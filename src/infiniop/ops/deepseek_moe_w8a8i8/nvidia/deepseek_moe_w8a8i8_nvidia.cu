#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "deepseek_moe_w8a8i8_nvidia.cuh"

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace op::deepseek_moe_w8a8i8::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

namespace {

constexpr size_t W8A8_GROUPED_GEMM_MIN_TOKENS = 2048;

constexpr size_t align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

size_t ptr_workspace_size(const DeepseekMoeW8A8I8Info &info) {
    return align_up(info.num_experts * sizeof(void *), 256) * 6;
}

struct RawLayout {
    size_t total_size;
    size_t hidden_packed_offset;
    size_t hidden_scale_offset;
    size_t intermediate_offset;
    size_t intermediate_packed_offset;
    size_t intermediate_scale_offset;
};

RawLayout make_raw_layout(const DeepseekMoeW8A8I8Info &info, size_t dtype_size) {
    RawLayout layout{};
    const size_t routes = info.ntokens * info.topk;
    size_t offset = ptr_workspace_size(info);

    layout.hidden_packed_offset = offset;
    offset = align_up(offset + info.ntokens * info.hidden_size * sizeof(int8_t), 256);
    layout.hidden_scale_offset = offset;
    offset = align_up(offset + info.ntokens * sizeof(float), 256);
    layout.intermediate_offset = offset;
    offset = align_up(offset + routes * info.intermediate_size * dtype_size, 256);
    layout.intermediate_packed_offset = offset;
    offset = align_up(offset + routes * info.intermediate_size * sizeof(int8_t), 256);
    layout.intermediate_scale_offset = offset;
    offset = align_up(offset + routes * sizeof(float), 256);
    layout.total_size = offset;
    return layout;
}

size_t raw_workspace_size(const DeepseekMoeW8A8I8Info &info, size_t dtype_size) {
    return make_raw_layout(info, dtype_size).total_size;
}

struct GroupedLayout {
    size_t total_size;
    size_t offsets_offset;
    size_t cursors_offset;
    size_t sorted_routes_offset;
    size_t route_to_sorted_offset;
    size_t hidden_packed_offset;
    size_t hidden_scale_offset;
    size_t sorted_hidden_packed_offset;
    size_t gate_i32_offset;
    size_t up_i32_offset;
    size_t intermediate_offset;
    size_t intermediate_packed_offset;
    size_t intermediate_scale_offset;
    size_t down_i32_offset;
};

GroupedLayout make_grouped_layout(const DeepseekMoeW8A8I8Info &info, size_t dtype_size) {
    GroupedLayout layout{};
    const size_t routes = info.ntokens * info.topk;
    size_t offset = ptr_workspace_size(info);

    layout.offsets_offset = offset;
    offset = align_up(offset + (info.num_experts + 1) * sizeof(int32_t), 256);
    layout.cursors_offset = offset;
    offset = align_up(offset + info.num_experts * sizeof(int32_t), 256);
    layout.sorted_routes_offset = offset;
    offset = align_up(offset + routes * sizeof(int32_t), 256);
    layout.route_to_sorted_offset = offset;
    offset = align_up(offset + routes * sizeof(int32_t), 256);
    layout.hidden_packed_offset = offset;
    offset = align_up(offset + info.ntokens * info.hidden_size * sizeof(int8_t), 256);
    layout.hidden_scale_offset = offset;
    offset = align_up(offset + info.ntokens * sizeof(float), 256);
    layout.sorted_hidden_packed_offset = offset;
    offset = align_up(offset + routes * info.hidden_size * sizeof(int8_t), 256);
    layout.gate_i32_offset = offset;
    offset = align_up(offset + routes * info.intermediate_size * sizeof(int32_t), 256);
    layout.up_i32_offset = offset;
    offset = align_up(offset + routes * info.intermediate_size * sizeof(int32_t), 256);
    layout.intermediate_offset = offset;
    offset = align_up(offset + routes * info.intermediate_size * dtype_size, 256);
    layout.intermediate_packed_offset = offset;
    offset = align_up(offset + routes * info.intermediate_size * sizeof(int8_t), 256);
    layout.intermediate_scale_offset = offset;
    offset = align_up(offset + routes * sizeof(float), 256);
    layout.down_i32_offset = offset;
    offset = align_up(offset + routes * info.hidden_size * sizeof(int32_t), 256);
    layout.total_size = offset;
    return layout;
}

template <typename T>
__device__ float to_float(T value) {
    return static_cast<float>(value);
}

template <>
__device__ float to_float<half>(half value) {
    return __half2float(value);
}

template <>
__device__ float to_float<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T>
__device__ T from_float(float value) {
    return static_cast<T>(value);
}

template <>
__device__ half from_float<half>(float value) {
    return __float2half_rn(value);
}

template <>
__device__ __nv_bfloat16 from_float<__nv_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}

__device__ int round_half_away_from_zero(float x) {
    const float ax = fabsf(x);
    const int r = static_cast<int>(floorf(ax + 0.5f));
    return x < 0.0f ? -r : r;
}

__device__ int8_t quantize_sym(float value, float scale) {
    if (scale == 0.0f) {
        return 0;
    }
    int q = round_half_away_from_zero(value / scale);
    if (q > 127) {
        q = 127;
    }
    if (q < -127) {
        q = -127;
    }
    return static_cast<int8_t>(q);
}

template <typename T>
__global__ void quantize_rows_kernel(
    int8_t *quantized,
    float *scales,
    const T *input,
    size_t rows,
    size_t cols) {
    const size_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const T *src = input + row * cols;
    int8_t *dst = quantized + row * cols;

    float local_max = 0.0f;
    for (size_t col = threadIdx.x; col < cols; col += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(to_float<T>(src[col])));
    }

    __shared__ float shared_max[256];
    shared_max[threadIdx.x] = local_max;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_max[threadIdx.x] = fmaxf(shared_max[threadIdx.x], shared_max[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    const float scale = shared_max[0] / 127.0f;
    if (threadIdx.x == 0) {
        scales[row] = scale;
    }
    for (size_t col = threadIdx.x; col < cols; col += blockDim.x) {
        dst[col] = quantize_sym(to_float<T>(src[col]), scale);
    }
}


__global__ void build_grouped_routes_kernel(
    const int *topk_indices,
    int32_t *offsets,
    int32_t *cursors,
    int32_t *sorted_routes,
    int32_t *route_to_sorted,
    size_t routes,
    size_t num_experts) {
    const size_t tid = threadIdx.x;
    const size_t stride = blockDim.x;

    for (size_t expert = tid; expert < num_experts; expert += stride) {
        cursors[expert] = 0;
    }
    for (size_t route = tid; route < routes; route += stride) {
        route_to_sorted[route] = -1;
    }
    __syncthreads();

    for (size_t route = tid; route < routes; route += stride) {
        const int expert = topk_indices[route];
        if (expert >= 0 && static_cast<size_t>(expert) < num_experts) {
            atomicAdd(cursors + expert, 1);
        }
    }
    __syncthreads();

    if (tid == 0) {
        int32_t running = 0;
        for (size_t expert = 0; expert < num_experts; ++expert) {
            const int32_t count = cursors[expert];
            offsets[expert] = running;
            cursors[expert] = running;
            running += count;
        }
        offsets[num_experts] = running;
    }
    __syncthreads();

    for (size_t route = tid; route < routes; route += stride) {
        const int expert = topk_indices[route];
        if (expert >= 0 && static_cast<size_t>(expert) < num_experts) {
            const int32_t sorted = atomicAdd(cursors + expert, 1);
            sorted_routes[sorted] = static_cast<int32_t>(route);
            route_to_sorted[route] = sorted;
        }
    }
}

__global__ void gather_sorted_hidden_kernel(
    int8_t *sorted_hidden_packed,
    const int8_t *hidden_packed,
    const int32_t *sorted_routes,
    size_t routes,
    size_t topk,
    size_t hidden_size) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = routes * hidden_size;
    if (idx >= total) {
        return;
    }
    const size_t sorted_route = idx / hidden_size;
    const size_t h = idx - sorted_route * hidden_size;
    const int32_t route = sorted_routes[sorted_route];
    const size_t token = static_cast<size_t>(route) / topk;
    sorted_hidden_packed[idx] = hidden_packed[token * hidden_size + h];
}

template <typename T>
__global__ void gate_up_grouped_post_kernel(
    T *intermediate,
    const int32_t *gate_i32,
    const int32_t *up_i32,
    const float *hidden_scales,
    const int32_t *sorted_routes,
    const int *topk_indices,
    const void *const *gate_weight_scales,
    const void *const *up_weight_scales,
    size_t routes,
    size_t topk,
    size_t intermediate_size) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = routes * intermediate_size;
    if (idx >= total) {
        return;
    }
    const size_t sorted_route = idx / intermediate_size;
    const size_t j = idx - sorted_route * intermediate_size;
    const int32_t route = sorted_routes[sorted_route];
    if (route < 0) {
        return;
    }
    const int expert = topk_indices[route];
    const size_t token = static_cast<size_t>(route) / topk;
    const float x_scale = hidden_scales[token];
    const float *gate_scale = reinterpret_cast<const float *>(gate_weight_scales[expert]);
    const float *up_scale = reinterpret_cast<const float *>(up_weight_scales[expert]);
    const float g = x_scale * gate_scale[j] * static_cast<float>(gate_i32[idx]);
    const float u = x_scale * up_scale[j] * static_cast<float>(up_i32[idx]);
    const float silu = g / (1.0f + __expf(-g));
    intermediate[idx] = from_float<T>(silu * u);
}


template <typename T>
__global__ void gate_up_grouped_post_quant_kernel(
    int8_t *intermediate_packed,
    float *intermediate_scales,
    const int32_t *gate_i32,
    const int32_t *up_i32,
    const float *hidden_scales,
    const int32_t *sorted_routes,
    const int *topk_indices,
    const void *const *gate_weight_scales,
    const void *const *up_weight_scales,
    size_t routes,
    size_t topk,
    size_t intermediate_size) {
    const size_t sorted_route = blockIdx.x;
    if (sorted_route >= routes) {
        return;
    }
    const int32_t route_i32 = sorted_routes[sorted_route];
    if (route_i32 < 0) {
        return;
    }
    const size_t route = static_cast<size_t>(route_i32);
    const int expert = topk_indices[route];
    const size_t token = route / topk;
    const float x_scale = hidden_scales[token];
    const float *gate_scale = reinterpret_cast<const float *>(gate_weight_scales[expert]);
    const float *up_scale = reinterpret_cast<const float *>(up_weight_scales[expert]);
    const size_t base = sorted_route * intermediate_size;

    __shared__ float shared_max[256];
    float local_max = 0.0f;
    for (size_t j = threadIdx.x; j < intermediate_size; j += blockDim.x) {
        const float g = x_scale * gate_scale[j] * static_cast<float>(gate_i32[base + j]);
        const float u = x_scale * up_scale[j] * static_cast<float>(up_i32[base + j]);
        const float silu = g / (1.0f + __expf(-g));
        local_max = fmaxf(local_max, fabsf(silu * u));
    }

    shared_max[threadIdx.x] = local_max;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_max[threadIdx.x] = fmaxf(shared_max[threadIdx.x], shared_max[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    const float scale = shared_max[0] / 127.0f;
    if (threadIdx.x == 0) {
        intermediate_scales[sorted_route] = scale;
    }

    for (size_t j = threadIdx.x; j < intermediate_size; j += blockDim.x) {
        const float g = x_scale * gate_scale[j] * static_cast<float>(gate_i32[base + j]);
        const float u = x_scale * up_scale[j] * static_cast<float>(up_i32[base + j]);
        const float silu = g / (1.0f + __expf(-g));
        intermediate_packed[base + j] = quantize_sym(silu * u, scale);
    }
}

template <typename T>
__global__ void down_grouped_post_kernel(
    T *out,
    const int32_t *down_i32,
    const float *intermediate_scales,
    const int32_t *route_to_sorted,
    const int *topk_indices,
    const float *topk_weights,
    const void *const *down_weight_scales,
    size_t ntokens,
    size_t hidden_size,
    size_t topk) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = ntokens * hidden_size;
    if (idx >= total) {
        return;
    }
    const size_t token = idx / hidden_size;
    const size_t h = idx - token * hidden_size;
    float acc = 0.0f;
    const size_t route_base = token * topk;
    for (size_t k = 0; k < topk; ++k) {
        const size_t route = route_base + k;
        const int32_t sorted_route = route_to_sorted[route];
        if (sorted_route < 0) {
            continue;
        }
        const int expert = topk_indices[route];
        const float *down_scale = reinterpret_cast<const float *>(down_weight_scales[expert]);
        acc += intermediate_scales[sorted_route] * down_scale[h]
             * static_cast<float>(down_i32[static_cast<size_t>(sorted_route) * hidden_size + h])
             * topk_weights[route];
    }
    out[idx] = from_float<T>(acc);
}


template <typename T>
__global__ void gate_up_post_kernel(
    T *intermediate,
    const int32_t *gate_i32,
    const int32_t *up_i32,
    const float *hidden_scales,
    const int *topk_indices,
    const void *const *gate_weight_scales,
    const void *const *up_weight_scales,
    size_t routes,
    size_t topk,
    size_t intermediate_size) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = routes * intermediate_size;
    if (idx >= total) {
        return;
    }
    const size_t route = idx / intermediate_size;
    const size_t j = idx - route * intermediate_size;
    const int expert = topk_indices[route];
    const size_t token = route / topk;
    const float x_scale = hidden_scales[token];
    const float *gate_scale = reinterpret_cast<const float *>(gate_weight_scales[expert]);
    const float *up_scale = reinterpret_cast<const float *>(up_weight_scales[expert]);
    const float g = x_scale * gate_scale[j] * static_cast<float>(gate_i32[idx]);
    const float u = x_scale * up_scale[j] * static_cast<float>(up_i32[idx]);
    const float silu = g / (1.0f + __expf(-g));
    intermediate[idx] = from_float<T>(silu * u);
}

template <typename T>
__global__ void down_post_kernel(
    T *out,
    const int32_t *down_i32,
    const float *intermediate_scales,
    const int *topk_indices,
    const float *topk_weights,
    const void *const *down_weight_scales,
    size_t ntokens,
    size_t hidden_size,
    size_t topk) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = ntokens * hidden_size;
    if (idx >= total) {
        return;
    }
    const size_t token = idx / hidden_size;
    const size_t h = idx - token * hidden_size;
    float acc = 0.0f;
    const size_t route_base = token * topk;
    for (size_t k = 0; k < topk; ++k) {
        const size_t route = route_base + k;
        const int expert = topk_indices[route];
        const float *down_scale = reinterpret_cast<const float *>(down_weight_scales[expert]);
        acc += intermediate_scales[route] * down_scale[h]
             * static_cast<float>(down_i32[route * hidden_size + h])
             * topk_weights[route];
    }
    out[idx] = from_float<T>(acc);
}

template <typename T>
__global__ void gate_up_w8a8i8_kernel(
    T *intermediate,
    const T *hidden,
    const int *topk_indices,
    const float *topk_weights,
    const void *const *gate_weights,
    const void *const *up_weights,
    const void *const *gate_weight_scales,
    const void *const *up_weight_scales,
    size_t ntokens,
    size_t hidden_size,
    size_t topk,
    size_t intermediate_size,
    size_t num_experts) {

    const size_t route = blockIdx.x / intermediate_size;
    const size_t j = blockIdx.x - route * intermediate_size;
    if (route >= ntokens * topk) {
        return;
    }
    const int expert = topk_indices[route];
    if (expert < 0 || static_cast<size_t>(expert) >= num_experts) {
        return;
    }
    const size_t token = route / topk;
    const T *x = hidden + token * hidden_size;
    const int8_t *gate = reinterpret_cast<const int8_t *>(gate_weights[expert]) + j * hidden_size;
    const int8_t *up = reinterpret_cast<const int8_t *>(up_weights[expert]) + j * hidden_size;
    const float *gate_scale = reinterpret_cast<const float *>(gate_weight_scales[expert]);
    const float *up_scale = reinterpret_cast<const float *>(up_weight_scales[expert]);

    float local_max = 0.0f;
    for (size_t h = threadIdx.x; h < hidden_size; h += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(to_float<T>(x[h])));
    }

    __shared__ float shared_max[256];
    __shared__ int gate_shared[256];
    __shared__ int up_shared[256];
    shared_max[threadIdx.x] = local_max;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_max[threadIdx.x] = fmaxf(shared_max[threadIdx.x], shared_max[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    const float x_scale = shared_max[0] / 127.0f;
    int gate_sum = 0;
    int up_sum = 0;
    for (size_t h = threadIdx.x; h < hidden_size; h += blockDim.x) {
        const int xq = static_cast<int>(quantize_sym(to_float<T>(x[h]), x_scale));
        gate_sum += xq * static_cast<int>(gate[h]);
        up_sum += xq * static_cast<int>(up[h]);
    }

    gate_shared[threadIdx.x] = gate_sum;
    up_shared[threadIdx.x] = up_sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            gate_shared[threadIdx.x] += gate_shared[threadIdx.x + stride];
            up_shared[threadIdx.x] += up_shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const float g = x_scale * gate_scale[j] * static_cast<float>(gate_shared[0]);
        const float u = x_scale * up_scale[j] * static_cast<float>(up_shared[0]);
        const float silu = g / (1.0f + __expf(-g));
        intermediate[route * intermediate_size + j] = from_float<T>(silu * u);
    }
}

template <typename T>
__global__ void down_w8a8i8_kernel(
    T *out,
    const T *intermediate,
    const int *topk_indices,
    const float *topk_weights,
    const void *const *down_weights,
    const void *const *down_weight_scales,
    size_t ntokens,
    size_t hidden_size,
    size_t topk,
    size_t intermediate_size,
    size_t num_experts) {

    const size_t linear = blockIdx.x;
    const size_t token = linear / hidden_size;
    const size_t h = linear - token * hidden_size;
    if (token >= ntokens) {
        return;
    }

    __shared__ float shared_max[256];
    __shared__ int shared_sum[256];
    float acc = 0.0f;
    const size_t route_base = token * topk;
    for (size_t k = 0; k < topk; ++k) {
        const size_t route = route_base + k;
        const int expert = topk_indices[route];
        if (expert < 0 || static_cast<size_t>(expert) >= num_experts) {
            continue;
        }

        const T *x = intermediate + route * intermediate_size;
        const int8_t *down = reinterpret_cast<const int8_t *>(down_weights[expert]) + h * intermediate_size;
        const float *down_scale = reinterpret_cast<const float *>(down_weight_scales[expert]);

        float local_max = 0.0f;
        for (size_t j = threadIdx.x; j < intermediate_size; j += blockDim.x) {
            local_max = fmaxf(local_max, fabsf(to_float<T>(x[j])));
        }

        shared_max[threadIdx.x] = local_max;
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                shared_max[threadIdx.x] = fmaxf(shared_max[threadIdx.x], shared_max[threadIdx.x + stride]);
            }
            __syncthreads();
        }

        const float x_scale = shared_max[0] / 127.0f;
        int sum = 0;
        for (size_t j = threadIdx.x; j < intermediate_size; j += blockDim.x) {
            const int xq = static_cast<int>(quantize_sym(to_float<T>(x[j]), x_scale));
            sum += xq * static_cast<int>(down[j]);
        }

        shared_sum[threadIdx.x] = sum;
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            acc += x_scale * down_scale[h] * static_cast<float>(shared_sum[0]) * topk_weights[route];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out[token * hidden_size + h] = from_float<T>(acc);
    }
}


template <typename T>
__global__ void gate_up_w8a8i8_packed_kernel(
    T *intermediate,
    const int8_t *hidden_packed,
    const float *hidden_scales,
    const int *topk_indices,
    const void *const *gate_weights,
    const void *const *up_weights,
    const void *const *gate_weight_scales,
    const void *const *up_weight_scales,
    size_t ntokens,
    size_t hidden_size,
    size_t topk,
    size_t intermediate_size,
    size_t num_experts) {

    const size_t route = blockIdx.x / intermediate_size;
    const size_t j = blockIdx.x - route * intermediate_size;
    if (route >= ntokens * topk) {
        return;
    }
    const int expert = topk_indices[route];
    if (expert < 0 || static_cast<size_t>(expert) >= num_experts) {
        return;
    }

    const size_t token = route / topk;
    const int8_t *x = hidden_packed + token * hidden_size;
    const float x_scale = hidden_scales[token];
    const int8_t *gate = reinterpret_cast<const int8_t *>(gate_weights[expert]) + j * hidden_size;
    const int8_t *up = reinterpret_cast<const int8_t *>(up_weights[expert]) + j * hidden_size;
    const float *gate_scale = reinterpret_cast<const float *>(gate_weight_scales[expert]);
    const float *up_scale = reinterpret_cast<const float *>(up_weight_scales[expert]);

    __shared__ int gate_shared[256];
    __shared__ int up_shared[256];
    int gate_sum = 0;
    int up_sum = 0;
    for (size_t h = threadIdx.x; h < hidden_size; h += blockDim.x) {
        const int xq = static_cast<int>(x[h]);
        gate_sum += xq * static_cast<int>(gate[h]);
        up_sum += xq * static_cast<int>(up[h]);
    }

    gate_shared[threadIdx.x] = gate_sum;
    up_shared[threadIdx.x] = up_sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            gate_shared[threadIdx.x] += gate_shared[threadIdx.x + stride];
            up_shared[threadIdx.x] += up_shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const float g = x_scale * gate_scale[j] * static_cast<float>(gate_shared[0]);
        const float u = x_scale * up_scale[j] * static_cast<float>(up_shared[0]);
        const float silu = g / (1.0f + __expf(-g));
        intermediate[route * intermediate_size + j] = from_float<T>(silu * u);
    }
}

template <typename T>
__global__ void down_w8a8i8_packed_kernel(
    T *out,
    const int8_t *intermediate_packed,
    const float *intermediate_scales,
    const int *topk_indices,
    const float *topk_weights,
    const void *const *down_weights,
    const void *const *down_weight_scales,
    size_t ntokens,
    size_t hidden_size,
    size_t topk,
    size_t intermediate_size,
    size_t num_experts) {

    const size_t linear = blockIdx.x;
    const size_t token = linear / hidden_size;
    const size_t h = linear - token * hidden_size;
    if (token >= ntokens) {
        return;
    }

    __shared__ int shared_sum[256];
    float acc = 0.0f;
    const size_t route_base = token * topk;
    for (size_t k = 0; k < topk; ++k) {
        const size_t route = route_base + k;
        const int expert = topk_indices[route];
        if (expert < 0 || static_cast<size_t>(expert) >= num_experts) {
            continue;
        }

        const int8_t *x = intermediate_packed + route * intermediate_size;
        const float x_scale = intermediate_scales[route];
        const int8_t *down = reinterpret_cast<const int8_t *>(down_weights[expert]) + h * intermediate_size;
        const float *down_scale = reinterpret_cast<const float *>(down_weight_scales[expert]);

        int sum = 0;
        for (size_t j = threadIdx.x; j < intermediate_size; j += blockDim.x) {
            sum += static_cast<int>(x[j]) * static_cast<int>(down[j]);
        }

        shared_sum[threadIdx.x] = sum;
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            acc += x_scale * down_scale[h] * static_cast<float>(shared_sum[0]) * topk_weights[route];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out[token * hidden_size + h] = from_float<T>(acc);
    }
}


template <typename T>
__global__ void quantize_rows_4096_kernel(
    int8_t *quantized,
    float *scales,
    const T *input,
    size_t rows) {
    const size_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    constexpr size_t COLS = 4096;
    const T *src = input + row * COLS;
    int8_t *dst = quantized + row * COLS;

    float local_max = 0.0f;
    for (size_t col = threadIdx.x; col < COLS; col += 256) {
        local_max = fmaxf(local_max, fabsf(to_float<T>(src[col])));
    }

    __shared__ float shared_max[256];
    shared_max[threadIdx.x] = local_max;
    __syncthreads();
    for (unsigned int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_max[threadIdx.x] = fmaxf(shared_max[threadIdx.x], shared_max[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    const float scale = shared_max[0] / 127.0f;
    if (threadIdx.x == 0) {
        scales[row] = scale;
    }
    for (size_t col = threadIdx.x; col < COLS; col += 256) {
        dst[col] = quantize_sym(to_float<T>(src[col]), scale);
    }
}

template <typename T>
__global__ void quantize_rows_2048_kernel(
    int8_t *quantized,
    float *scales,
    const T *input,
    size_t rows) {
    const size_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    constexpr size_t COLS = 2048;
    const T *src = input + row * COLS;
    int8_t *dst = quantized + row * COLS;

    float local_max = 0.0f;
    for (size_t col = threadIdx.x; col < COLS; col += 256) {
        local_max = fmaxf(local_max, fabsf(to_float<T>(src[col])));
    }

    __shared__ float shared_max[256];
    shared_max[threadIdx.x] = local_max;
    __syncthreads();
    for (unsigned int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_max[threadIdx.x] = fmaxf(shared_max[threadIdx.x], shared_max[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    const float scale = shared_max[0] / 127.0f;
    if (threadIdx.x == 0) {
        scales[row] = scale;
    }
    for (size_t col = threadIdx.x; col < COLS; col += 256) {
        dst[col] = quantize_sym(to_float<T>(src[col]), scale);
    }
}

template <typename T>
__global__ void gate_up_w8a8i8_packed_h4096_kernel(
    T *intermediate,
    const int8_t *hidden_packed,
    const float *hidden_scales,
    const int *topk_indices,
    const void *const *gate_weights,
    const void *const *up_weights,
    const void *const *gate_weight_scales,
    const void *const *up_weight_scales,
    size_t routes,
    size_t topk,
    size_t num_experts) {
    constexpr size_t HIDDEN = 4096;
    constexpr size_t INTERMEDIATE = 2048;
    const size_t block = blockIdx.x;
    const size_t route = block / INTERMEDIATE;
    const size_t j = block - route * INTERMEDIATE;
    if (route >= routes) {
        return;
    }
    const int expert = topk_indices[route];
    if (expert < 0 || static_cast<size_t>(expert) >= num_experts) {
        return;
    }

    const size_t token = route / topk;
    const int8_t *x = hidden_packed + token * HIDDEN;
    const float x_scale = hidden_scales[token];
    const int8_t *gate = reinterpret_cast<const int8_t *>(gate_weights[expert]) + j * HIDDEN;
    const int8_t *up = reinterpret_cast<const int8_t *>(up_weights[expert]) + j * HIDDEN;
    const float *gate_scale = reinterpret_cast<const float *>(gate_weight_scales[expert]);
    const float *up_scale = reinterpret_cast<const float *>(up_weight_scales[expert]);

    __shared__ int gate_shared[256];
    __shared__ int up_shared[256];
    int gate_sum = 0;
    int up_sum = 0;
    for (size_t h = threadIdx.x; h < HIDDEN; h += 256) {
        const int xq = static_cast<int>(x[h]);
        gate_sum += xq * static_cast<int>(gate[h]);
        up_sum += xq * static_cast<int>(up[h]);
    }

    gate_shared[threadIdx.x] = gate_sum;
    up_shared[threadIdx.x] = up_sum;
    __syncthreads();
    for (unsigned int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            gate_shared[threadIdx.x] += gate_shared[threadIdx.x + stride];
            up_shared[threadIdx.x] += up_shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const float g = x_scale * gate_scale[j] * static_cast<float>(gate_shared[0]);
        const float u = x_scale * up_scale[j] * static_cast<float>(up_shared[0]);
        const float silu = g / (1.0f + __expf(-g));
        intermediate[route * INTERMEDIATE + j] = from_float<T>(silu * u);
    }
}

template <typename T>
__global__ void down_w8a8i8_packed_h2048_topk6_kernel(
    T *out,
    const int8_t *intermediate_packed,
    const float *intermediate_scales,
    const int *topk_indices,
    const float *topk_weights,
    const void *const *down_weights,
    const void *const *down_weight_scales,
    size_t ntokens,
    size_t hidden_size,
    size_t num_experts) {
    constexpr size_t TOPK = 6;
    constexpr size_t INTERMEDIATE = 2048;
    const size_t linear = blockIdx.x;
    const size_t token = linear / hidden_size;
    const size_t h = linear - token * hidden_size;
    if (token >= ntokens) {
        return;
    }

    __shared__ int shared_sum[256];
    float acc = 0.0f;
    const size_t route_base = token * TOPK;
    for (size_t k = 0; k < TOPK; ++k) {
        const size_t route = route_base + k;
        const int expert = topk_indices[route];
        if (expert >= 0 && static_cast<size_t>(expert) < num_experts) {
            const int8_t *x = intermediate_packed + route * INTERMEDIATE;
            const float x_scale = intermediate_scales[route];
            const int8_t *down = reinterpret_cast<const int8_t *>(down_weights[expert]) + h * INTERMEDIATE;
            const float *down_scale = reinterpret_cast<const float *>(down_weight_scales[expert]);

            int sum = 0;
            for (size_t j = threadIdx.x; j < INTERMEDIATE; j += 256) {
                sum += static_cast<int>(x[j]) * static_cast<int>(down[j]);
            }

            shared_sum[threadIdx.x] = sum;
            __syncthreads();
            for (unsigned int stride = 128; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride) {
                    shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
                }
                __syncthreads();
            }

            if (threadIdx.x == 0) {
                acc += x_scale * down_scale[h] * static_cast<float>(shared_sum[0]) * topk_weights[route];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out[token * hidden_size + h] = from_float<T>(acc);
    }
}


infiniStatus_t launch_single_i8_gemm(
    const device::nvidia::Handle::Internal *internal,
    cudaStream_t stream,
    const void *a,
    const void *b,
    void *c,
    int m_rows,
    int k,
    int n_cols) {
    const int32_t alpha = 1;
    const int32_t beta = 0;
    const int lda = k;
    const int ldb = k;
    const int ldc = n_cols;
    CHECK_STATUS(internal->useCublas(
        stream,
        [&](cublasHandle_t handle) {
            CHECK_CUBLAS(cublasGemmEx(
                handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                n_cols,
                m_rows,
                k,
                &alpha,
                a,
                CUDA_R_8I,
                lda,
                b,
                CUDA_R_8I,
                ldb,
                &beta,
                c,
                CUDA_R_32I,
                ldc,
                CUBLAS_COMPUTE_32I,
                CUBLAS_GEMM_DEFAULT));
            return INFINI_STATUS_SUCCESS;
        }));
    return INFINI_STATUS_SUCCESS;
}


template <typename T>
infiniStatus_t launch_grouped_typed(
    void *workspace,
    size_t workspace_size,
    const DeepseekMoeW8A8I8Info &info,
    void *out,
    const void *hidden,
    const void *topk_indices,
    const void *topk_weights,
    const void *const *gate_ptrs,
    const void *const *up_ptrs,
    const void *const *down_ptrs,
    const void *const *gate_scale_ptrs,
    const void *const *up_scale_ptrs,
    const void *const *down_scale_ptrs,
    cudaStream_t stream,
    const device::nvidia::Handle::Internal *internal) {
    const size_t routes = info.ntokens * info.topk;
    const auto layout = make_grouped_layout(info, sizeof(T));
    if (workspace_size < layout.total_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    auto *base = reinterpret_cast<char *>(workspace);
    auto *offsets = reinterpret_cast<int32_t *>(base + layout.offsets_offset);
    auto *cursors = reinterpret_cast<int32_t *>(base + layout.cursors_offset);
    auto *sorted_routes = reinterpret_cast<int32_t *>(base + layout.sorted_routes_offset);
    auto *route_to_sorted = reinterpret_cast<int32_t *>(base + layout.route_to_sorted_offset);
    auto *hidden_packed = reinterpret_cast<int8_t *>(base + layout.hidden_packed_offset);
    auto *hidden_scales = reinterpret_cast<float *>(base + layout.hidden_scale_offset);
    auto *sorted_hidden_packed = reinterpret_cast<int8_t *>(base + layout.sorted_hidden_packed_offset);
    auto *gate_i32 = reinterpret_cast<int32_t *>(base + layout.gate_i32_offset);
    auto *up_i32 = reinterpret_cast<int32_t *>(base + layout.up_i32_offset);
    auto *intermediate = reinterpret_cast<T *>(base + layout.intermediate_offset);
    auto *intermediate_packed = reinterpret_cast<int8_t *>(base + layout.intermediate_packed_offset);
    auto *intermediate_scales = reinterpret_cast<float *>(base + layout.intermediate_scale_offset);
    auto *down_i32 = reinterpret_cast<int32_t *>(base + layout.down_i32_offset);

    constexpr int threads = 256;
    quantize_rows_kernel<T><<<static_cast<unsigned int>(info.ntokens), threads, 0, stream>>>(
        hidden_packed,
        hidden_scales,
        reinterpret_cast<const T *>(hidden),
        info.ntokens,
        info.hidden_size);
    CHECK_CUDA(cudaGetLastError());

    build_grouped_routes_kernel<<<1, threads, 0, stream>>>(
        reinterpret_cast<const int *>(topk_indices),
        offsets,
        cursors,
        sorted_routes,
        route_to_sorted,
        routes,
        info.num_experts);
    CHECK_CUDA(cudaGetLastError());

    std::vector<int32_t> offsets_host(info.num_experts + 1);
    std::vector<const void *> gate_host(info.num_experts);
    std::vector<const void *> up_host(info.num_experts);
    std::vector<const void *> down_host(info.num_experts);
    CHECK_CUDA(cudaMemcpyAsync(offsets_host.data(), offsets, (info.num_experts + 1) * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaMemcpyAsync(gate_host.data(), gate_ptrs, info.num_experts * sizeof(void *), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaMemcpyAsync(up_host.data(), up_ptrs, info.num_experts * sizeof(void *), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaMemcpyAsync(down_host.data(), down_ptrs, info.num_experts * sizeof(void *), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    const size_t hidden_total = routes * info.hidden_size;
    gather_sorted_hidden_kernel<<<static_cast<unsigned int>((hidden_total + threads - 1) / threads), threads, 0, stream>>>(
        sorted_hidden_packed,
        hidden_packed,
        sorted_routes,
        routes,
        info.topk,
        info.hidden_size);
    CHECK_CUDA(cudaGetLastError());

    for (size_t expert = 0; expert < info.num_experts; ++expert) {
        const int32_t begin = offsets_host[expert];
        const int32_t end = offsets_host[expert + 1];
        const int32_t count = end - begin;
        if (count <= 0) {
            continue;
        }
        CHECK_STATUS(launch_single_i8_gemm(internal, stream, gate_host[expert], sorted_hidden_packed + static_cast<size_t>(begin) * info.hidden_size,
                                           gate_i32 + static_cast<size_t>(begin) * info.intermediate_size, count,
                                           static_cast<int>(info.hidden_size), static_cast<int>(info.intermediate_size)));
        CHECK_STATUS(launch_single_i8_gemm(internal, stream, up_host[expert], sorted_hidden_packed + static_cast<size_t>(begin) * info.hidden_size,
                                           up_i32 + static_cast<size_t>(begin) * info.intermediate_size, count,
                                           static_cast<int>(info.hidden_size), static_cast<int>(info.intermediate_size)));
    }

    const bool use_legacy_grouped_post = std::getenv("INFINICORE_DSV4_W8A8_GROUPED_LEGACY_POST") != nullptr;
    if (use_legacy_grouped_post) {
        const size_t gate_post_total = routes * info.intermediate_size;
        gate_up_grouped_post_kernel<T><<<static_cast<unsigned int>((gate_post_total + threads - 1) / threads), threads, 0, stream>>>(
            intermediate,
            gate_i32,
            up_i32,
            hidden_scales,
            sorted_routes,
            reinterpret_cast<const int *>(topk_indices),
            gate_scale_ptrs,
            up_scale_ptrs,
            routes,
            info.topk,
            info.intermediate_size);
        CHECK_CUDA(cudaGetLastError());

        quantize_rows_kernel<T><<<static_cast<unsigned int>(routes), threads, 0, stream>>>(
            intermediate_packed,
            intermediate_scales,
            intermediate,
            routes,
            info.intermediate_size);
        CHECK_CUDA(cudaGetLastError());
    } else {
        gate_up_grouped_post_quant_kernel<T><<<static_cast<unsigned int>(routes), threads, 0, stream>>>(
            intermediate_packed,
            intermediate_scales,
            gate_i32,
            up_i32,
            hidden_scales,
            sorted_routes,
            reinterpret_cast<const int *>(topk_indices),
            gate_scale_ptrs,
            up_scale_ptrs,
            routes,
            info.topk,
            info.intermediate_size);
        CHECK_CUDA(cudaGetLastError());
    }

    for (size_t expert = 0; expert < info.num_experts; ++expert) {
        const int32_t begin = offsets_host[expert];
        const int32_t end = offsets_host[expert + 1];
        const int32_t count = end - begin;
        if (count <= 0) {
            continue;
        }
        CHECK_STATUS(launch_single_i8_gemm(internal, stream, down_host[expert], intermediate_packed + static_cast<size_t>(begin) * info.intermediate_size,
                                           down_i32 + static_cast<size_t>(begin) * info.hidden_size, count,
                                           static_cast<int>(info.intermediate_size), static_cast<int>(info.hidden_size)));
    }

    const size_t down_post_total = info.ntokens * info.hidden_size;
    down_grouped_post_kernel<T><<<static_cast<unsigned int>((down_post_total + threads - 1) / threads), threads, 0, stream>>>(
        reinterpret_cast<T *>(out),
        down_i32,
        intermediate_scales,
        route_to_sorted,
        reinterpret_cast<const int *>(topk_indices),
        reinterpret_cast<const float *>(topk_weights),
        down_scale_ptrs,
        info.ntokens,
        info.hidden_size,
        info.topk);
    CHECK_CUDA(cudaGetLastError());

    return INFINI_STATUS_SUCCESS;
}

template <typename T>
infiniStatus_t launch_raw_typed(
    void *workspace,
    size_t workspace_size,
    const DeepseekMoeW8A8I8Info &info,
    void *out,
    const void *hidden,
    const void *topk_indices,
    const void *topk_weights,
    const void *const *gate_ptrs,
    const void *const *up_ptrs,
    const void *const *down_ptrs,
    const void *const *gate_scale_ptrs,
    const void *const *up_scale_ptrs,
    const void *const *down_scale_ptrs,
    cudaStream_t stream) {
    const auto layout = make_raw_layout(info, sizeof(T));
    if (workspace_size < layout.total_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    auto *base = reinterpret_cast<char *>(workspace);
    auto *hidden_packed = reinterpret_cast<int8_t *>(base + layout.hidden_packed_offset);
    auto *hidden_scales = reinterpret_cast<float *>(base + layout.hidden_scale_offset);
    auto *intermediate = reinterpret_cast<T *>(base + layout.intermediate_offset);
    auto *intermediate_packed = reinterpret_cast<int8_t *>(base + layout.intermediate_packed_offset);
    auto *intermediate_scales = reinterpret_cast<float *>(base + layout.intermediate_scale_offset);

    constexpr int threads = 256;
    const bool use_legacy_raw = std::getenv("INFINICORE_DSV4_W8A8_RAW_LEGACY") != nullptr;
    if (use_legacy_raw) {
        const dim3 gate_blocks(static_cast<unsigned int>(info.ntokens * info.topk * info.intermediate_size));
        gate_up_w8a8i8_kernel<T><<<gate_blocks, threads, 0, stream>>>(
            intermediate,
            reinterpret_cast<const T *>(hidden),
            reinterpret_cast<const int *>(topk_indices),
            reinterpret_cast<const float *>(topk_weights),
            gate_ptrs,
            up_ptrs,
            gate_scale_ptrs,
            up_scale_ptrs,
            info.ntokens,
            info.hidden_size,
            info.topk,
            info.intermediate_size,
            info.num_experts);
        CHECK_CUDA(cudaGetLastError());

        const dim3 down_blocks(static_cast<unsigned int>(info.ntokens * info.hidden_size));
        down_w8a8i8_kernel<T><<<down_blocks, threads, 0, stream>>>(
            reinterpret_cast<T *>(out),
            intermediate,
            reinterpret_cast<const int *>(topk_indices),
            reinterpret_cast<const float *>(topk_weights),
            down_ptrs,
            down_scale_ptrs,
            info.ntokens,
            info.hidden_size,
            info.topk,
            info.intermediate_size,
            info.num_experts);
        CHECK_CUDA(cudaGetLastError());
        return INFINI_STATUS_SUCCESS;
    }

    const bool use_dsv4_decode_fast_path = info.hidden_size == 4096
                                      && info.intermediate_size == 2048
                                      && info.topk == 6
                                      && std::getenv("INFINICORE_DSV4_W8A8_DISABLE_DECODE_FAST_PATH") == nullptr;
    if (use_dsv4_decode_fast_path) {
        quantize_rows_4096_kernel<T><<<static_cast<unsigned int>(info.ntokens), threads, 0, stream>>>(
            hidden_packed,
            hidden_scales,
            reinterpret_cast<const T *>(hidden),
            info.ntokens);
        CHECK_CUDA(cudaGetLastError());

        const size_t routes = info.ntokens * info.topk;
        const dim3 gate_blocks(static_cast<unsigned int>(routes * info.intermediate_size));
        gate_up_w8a8i8_packed_h4096_kernel<T><<<gate_blocks, threads, 0, stream>>>(
            intermediate,
            hidden_packed,
            hidden_scales,
            reinterpret_cast<const int *>(topk_indices),
            gate_ptrs,
            up_ptrs,
            gate_scale_ptrs,
            up_scale_ptrs,
            routes,
            info.topk,
            info.num_experts);
        CHECK_CUDA(cudaGetLastError());

        quantize_rows_2048_kernel<T><<<static_cast<unsigned int>(routes), threads, 0, stream>>>(
            intermediate_packed,
            intermediate_scales,
            intermediate,
            routes);
        CHECK_CUDA(cudaGetLastError());

        const dim3 down_blocks(static_cast<unsigned int>(info.ntokens * info.hidden_size));
        down_w8a8i8_packed_h2048_topk6_kernel<T><<<down_blocks, threads, 0, stream>>>(
            reinterpret_cast<T *>(out),
            intermediate_packed,
            intermediate_scales,
            reinterpret_cast<const int *>(topk_indices),
            reinterpret_cast<const float *>(topk_weights),
            down_ptrs,
            down_scale_ptrs,
            info.ntokens,
            info.hidden_size,
            info.num_experts);
        CHECK_CUDA(cudaGetLastError());
        return INFINI_STATUS_SUCCESS;
    }

    quantize_rows_kernel<T><<<static_cast<unsigned int>(info.ntokens), threads, 0, stream>>>(
        hidden_packed,
        hidden_scales,
        reinterpret_cast<const T *>(hidden),
        info.ntokens,
        info.hidden_size);
    CHECK_CUDA(cudaGetLastError());

    const dim3 gate_blocks(static_cast<unsigned int>(info.ntokens * info.topk * info.intermediate_size));
    gate_up_w8a8i8_packed_kernel<T><<<gate_blocks, threads, 0, stream>>>(
        intermediate,
        hidden_packed,
        hidden_scales,
        reinterpret_cast<const int *>(topk_indices),
        gate_ptrs,
        up_ptrs,
        gate_scale_ptrs,
        up_scale_ptrs,
        info.ntokens,
        info.hidden_size,
        info.topk,
        info.intermediate_size,
        info.num_experts);
    CHECK_CUDA(cudaGetLastError());

    quantize_rows_kernel<T><<<static_cast<unsigned int>(info.ntokens * info.topk), threads, 0, stream>>>(
        intermediate_packed,
        intermediate_scales,
        intermediate,
        info.ntokens * info.topk,
        info.intermediate_size);
    CHECK_CUDA(cudaGetLastError());

    const dim3 down_blocks(static_cast<unsigned int>(info.ntokens * info.hidden_size));
    down_w8a8i8_packed_kernel<T><<<down_blocks, threads, 0, stream>>>(
        reinterpret_cast<T *>(out),
        intermediate_packed,
        intermediate_scales,
        reinterpret_cast<const int *>(topk_indices),
        reinterpret_cast<const float *>(topk_weights),
        down_ptrs,
        down_scale_ptrs,
        info.ntokens,
        info.hidden_size,
        info.topk,
        info.intermediate_size,
        info.num_experts);
    CHECK_CUDA(cudaGetLastError());

    return INFINI_STATUS_SUCCESS;
}

template <typename T>
infiniStatus_t launch_typed(
    void *workspace,
    size_t workspace_size,
    const DeepseekMoeW8A8I8Info &info,
    void *out,
    const void *hidden,
    const void *topk_indices,
    const void *topk_weights,
    const void *const *gate_weights,
    const void *const *up_weights,
    const void *const *down_weights,
    const void *const *gate_weight_scales,
    const void *const *up_weight_scales,
    const void *const *down_weight_scales,
    cudaStream_t stream,
    bool ptrs_on_device,
    const device::nvidia::Handle::Internal *internal) {

    const size_t ptr_bytes = align_up(info.num_experts * sizeof(void *), 256);
    auto *base = reinterpret_cast<char *>(workspace);
    const void *const *gate_ptrs = reinterpret_cast<const void *const *>(base);
    const void *const *up_ptrs = reinterpret_cast<const void *const *>(base + ptr_bytes);
    const void *const *down_ptrs = reinterpret_cast<const void *const *>(base + ptr_bytes * 2);
    const void *const *gate_scale_ptrs = reinterpret_cast<const void *const *>(base + ptr_bytes * 3);
    const void *const *up_scale_ptrs = reinterpret_cast<const void *const *>(base + ptr_bytes * 4);
    const void *const *down_scale_ptrs = reinterpret_cast<const void *const *>(base + ptr_bytes * 5);

    if (ptrs_on_device) {
        gate_ptrs = gate_weights;
        up_ptrs = up_weights;
        down_ptrs = down_weights;
        gate_scale_ptrs = gate_weight_scales;
        up_scale_ptrs = up_weight_scales;
        down_scale_ptrs = down_weight_scales;
    } else {
        auto **gate_workspace = reinterpret_cast<const void **>(base);
        auto **up_workspace = reinterpret_cast<const void **>(base + ptr_bytes);
        auto **down_workspace = reinterpret_cast<const void **>(base + ptr_bytes * 2);
        auto **gate_scale_workspace = reinterpret_cast<const void **>(base + ptr_bytes * 3);
        auto **up_scale_workspace = reinterpret_cast<const void **>(base + ptr_bytes * 4);
        auto **down_scale_workspace = reinterpret_cast<const void **>(base + ptr_bytes * 5);
        CHECK_CUDA(cudaMemcpyAsync(gate_workspace, gate_weights, info.num_experts * sizeof(void *), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(up_workspace, up_weights, info.num_experts * sizeof(void *), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(down_workspace, down_weights, info.num_experts * sizeof(void *), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(gate_scale_workspace, gate_weight_scales, info.num_experts * sizeof(void *), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(up_scale_workspace, up_weight_scales, info.num_experts * sizeof(void *), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(down_scale_workspace, down_weight_scales, info.num_experts * sizeof(void *), cudaMemcpyHostToDevice, stream));
        gate_ptrs = gate_workspace;
        up_ptrs = up_workspace;
        down_ptrs = down_workspace;
        gate_scale_ptrs = gate_scale_workspace;
        up_scale_ptrs = up_scale_workspace;
        down_scale_ptrs = down_scale_workspace;
    }

    const char *grouped_env = std::getenv("INFINICORE_DSV4_W8A8_GROUPED_GEMM");
    const bool force_grouped = grouped_env != nullptr && std::string(grouped_env) == "1";
    const bool disable_grouped = grouped_env != nullptr && std::string(grouped_env) == "0";
    const bool auto_grouped = info.ntokens >= W8A8_GROUPED_GEMM_MIN_TOKENS && info.num_experts <= 256;
    const bool use_grouped_gemm = force_grouped || (!disable_grouped && auto_grouped);
    if (use_grouped_gemm && info.ntokens >= 4) {
        return launch_grouped_typed<T>(workspace, workspace_size, info, out, hidden, topk_indices, topk_weights,
                                       gate_ptrs, up_ptrs, down_ptrs, gate_scale_ptrs, up_scale_ptrs, down_scale_ptrs,
                                       stream, internal);
    }
    return launch_raw_typed<T>(workspace, workspace_size, info, out, hidden, topk_indices, topk_weights,
                               gate_ptrs, up_ptrs, down_ptrs, gate_scale_ptrs, up_scale_ptrs, down_scale_ptrs,
                               stream);
}

} // namespace

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t hidden_desc,
    infiniopTensorDescriptor_t topk_indices_desc,
    infiniopTensorDescriptor_t topk_weights_desc,
    size_t intermediate_size,
    size_t num_experts) {

    auto result = DeepseekMoeW8A8I8Info::create(out_desc, hidden_desc, topk_indices_desc, topk_weights_desc, intermediate_size, num_experts);
    CHECK_RESULT(result);
    auto info = result.take();

    const size_t dtype_size = info.dtype == INFINI_DTYPE_F16 ? sizeof(half) : sizeof(__nv_bfloat16);
    const size_t workspace_size = std::max(raw_workspace_size(info, dtype_size),
                                           make_grouped_layout(info, dtype_size).total_size);

    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        info,
        workspace_size,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *hidden,
    const void *topk_indices,
    const void *topk_weights,
    const void *const *gate_weights,
    const void *const *up_weights,
    const void *const *down_weights,
    const void *const *gate_weight_scales,
    const void *const *up_weight_scales,
    const void *const *down_weight_scales,
    void *stream_) const {

    auto stream = reinterpret_cast<cudaStream_t>(stream_);
    if (_info.dtype == INFINI_DTYPE_F16) {
        return launch_typed<half>(workspace, workspace_size, _info, out, hidden, topk_indices, topk_weights,
                                  gate_weights, up_weights, down_weights, gate_weight_scales, up_weight_scales,
                                  down_weight_scales, stream, false, _opaque->internal.get());
    }
    if (_info.dtype == INFINI_DTYPE_BF16) {
        return launch_typed<__nv_bfloat16>(workspace, workspace_size, _info, out, hidden, topk_indices, topk_weights,
                                           gate_weights, up_weights, down_weights, gate_weight_scales, up_weight_scales,
                                           down_weight_scales, stream, false, _opaque->internal.get());
    }
    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}

infiniStatus_t Descriptor::calculateWithDevicePtrs(
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *hidden,
    const void *topk_indices,
    const void *topk_weights,
    const void *gate_weight_ptrs,
    const void *up_weight_ptrs,
    const void *down_weight_ptrs,
    const void *gate_weight_scale_ptrs,
    const void *up_weight_scale_ptrs,
    const void *down_weight_scale_ptrs,
    void *stream_) const {

    auto stream = reinterpret_cast<cudaStream_t>(stream_);
    auto gate_weights = reinterpret_cast<const void *const *>(gate_weight_ptrs);
    auto up_weights = reinterpret_cast<const void *const *>(up_weight_ptrs);
    auto down_weights = reinterpret_cast<const void *const *>(down_weight_ptrs);
    auto gate_weight_scales = reinterpret_cast<const void *const *>(gate_weight_scale_ptrs);
    auto up_weight_scales = reinterpret_cast<const void *const *>(up_weight_scale_ptrs);
    auto down_weight_scales = reinterpret_cast<const void *const *>(down_weight_scale_ptrs);
    if (_info.dtype == INFINI_DTYPE_F16) {
        return launch_typed<half>(workspace, workspace_size, _info, out, hidden, topk_indices, topk_weights,
                                  gate_weights, up_weights, down_weights, gate_weight_scales, up_weight_scales,
                                  down_weight_scales, stream, true, _opaque->internal.get());
    }
    if (_info.dtype == INFINI_DTYPE_BF16) {
        return launch_typed<__nv_bfloat16>(workspace, workspace_size, _info, out, hidden, topk_indices, topk_weights,
                                           gate_weights, up_weights, down_weights, gate_weight_scales, up_weight_scales,
                                           down_weight_scales, stream, true, _opaque->internal.get());
    }
    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}

} // namespace op::deepseek_moe_w8a8i8::nvidia
