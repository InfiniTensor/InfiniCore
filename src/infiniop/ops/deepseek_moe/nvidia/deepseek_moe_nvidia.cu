#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "deepseek_moe_nvidia.cuh"

namespace op::deepseek_moe::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

namespace {

constexpr size_t align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
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

template <typename T>
__global__ void gate_up_kernel(
    T *intermediate,
    const T *hidden,
    const int *topk_indices,
    const float *topk_weights,
    const void *const *gate_weights,
    const void *const *up_weights,
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
    const T *gate = reinterpret_cast<const T *>(gate_weights[expert]) + j * hidden_size;
    const T *up = reinterpret_cast<const T *>(up_weights[expert]) + j * hidden_size;

    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (size_t h = threadIdx.x; h < hidden_size; h += blockDim.x) {
        const float xv = to_float<T>(x[h]);
        gate_sum += xv * to_float<T>(gate[h]);
        up_sum += xv * to_float<T>(up[h]);
    }

    __shared__ float gate_shared[256];
    __shared__ float up_shared[256];
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
        const float g = gate_shared[0];
        const float silu = g / (1.0f + __expf(-g));
        intermediate[route * intermediate_size + j] = from_float<T>(silu * up_shared[0] * topk_weights[route]);
    }
}

template <typename T>
__global__ void down_kernel(
    T *out,
    const T *intermediate,
    const int *topk_indices,
    const void *const *down_weights,
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

    float acc = 0.0f;
    const size_t route_base = token * topk;
    const size_t count = topk * intermediate_size;
    for (size_t idx = threadIdx.x; idx < count; idx += blockDim.x) {
        const size_t k = idx / intermediate_size;
        const size_t j = idx - k * intermediate_size;
        const size_t route = route_base + k;
        const int expert = topk_indices[route];
        if (expert >= 0 && static_cast<size_t>(expert) < num_experts) {
            const T *down = reinterpret_cast<const T *>(down_weights[expert]) + h * intermediate_size;
            acc += to_float<T>(intermediate[route * intermediate_size + j]) * to_float<T>(down[j]);
        }
    }

    __shared__ float shared[256];
    shared[threadIdx.x] = acc;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[token * hidden_size + h] = from_float<T>(shared[0]);
    }
}

template <typename T>
infiniStatus_t launch_typed(
    void *workspace,
    size_t workspace_size,
    const DeepseekMoeInfo &info,
    void *out,
    const void *hidden,
    const void *topk_indices,
    const void *topk_weights,
    const void *const *gate_weights,
    const void *const *up_weights,
    const void *const *down_weights,
    cudaStream_t stream,
    bool weight_ptrs_on_device) {

    const size_t ptr_bytes = align_up(info.num_experts * sizeof(void *), 256);
    const size_t ptr_workspace = ptr_bytes * 3;
    const size_t intermediate_offset = align_up(ptr_workspace, 256);
    const size_t intermediate_bytes = info.ntokens * info.topk * info.intermediate_size * sizeof(T);
    if (workspace_size < intermediate_offset + intermediate_bytes) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    auto *base = reinterpret_cast<char *>(workspace);
    const void *const *gate_ptrs = reinterpret_cast<const void *const *>(base);
    const void *const *up_ptrs = reinterpret_cast<const void *const *>(base + ptr_bytes);
    const void *const *down_ptrs = reinterpret_cast<const void *const *>(base + ptr_bytes * 2);
    auto *intermediate = reinterpret_cast<T *>(base + intermediate_offset);

    if (weight_ptrs_on_device) {
        gate_ptrs = gate_weights;
        up_ptrs = up_weights;
        down_ptrs = down_weights;
    } else {
        auto **gate_workspace = reinterpret_cast<const void **>(base);
        auto **up_workspace = reinterpret_cast<const void **>(base + ptr_bytes);
        auto **down_workspace = reinterpret_cast<const void **>(base + ptr_bytes * 2);
        CHECK_CUDA(cudaMemcpyAsync(gate_workspace, gate_weights, info.num_experts * sizeof(void *), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(up_workspace, up_weights, info.num_experts * sizeof(void *), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(down_workspace, down_weights, info.num_experts * sizeof(void *), cudaMemcpyHostToDevice, stream));
        gate_ptrs = gate_workspace;
        up_ptrs = up_workspace;
        down_ptrs = down_workspace;
    }

    constexpr int threads = 256;
    const dim3 gate_blocks(static_cast<unsigned int>(info.ntokens * info.topk * info.intermediate_size));
    gate_up_kernel<T><<<gate_blocks, threads, 0, stream>>>(
        intermediate,
        reinterpret_cast<const T *>(hidden),
        reinterpret_cast<const int *>(topk_indices),
        reinterpret_cast<const float *>(topk_weights),
        gate_ptrs,
        up_ptrs,
        info.ntokens,
        info.hidden_size,
        info.topk,
        info.intermediate_size,
        info.num_experts);

    const dim3 down_blocks(static_cast<unsigned int>(info.ntokens * info.hidden_size));
    down_kernel<T><<<down_blocks, threads, 0, stream>>>(
        reinterpret_cast<T *>(out),
        intermediate,
        reinterpret_cast<const int *>(topk_indices),
        down_ptrs,
        info.ntokens,
        info.hidden_size,
        info.topk,
        info.intermediate_size,
        info.num_experts);

    return INFINI_STATUS_SUCCESS;
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

    auto result = DeepseekMoeInfo::create(out_desc, hidden_desc, topk_indices_desc, topk_weights_desc, intermediate_size, num_experts);
    CHECK_RESULT(result);
    auto info = result.take();

    const size_t dtype_size = info.dtype == INFINI_DTYPE_F16 ? sizeof(half) : sizeof(__nv_bfloat16);
    const size_t ptr_bytes = align_up(info.num_experts * sizeof(void *), 256);
    const size_t intermediate_offset = align_up(ptr_bytes * 3, 256);
    const size_t intermediate_bytes = info.ntokens * info.topk * info.intermediate_size * dtype_size;
    const size_t workspace_size = intermediate_offset + intermediate_bytes;

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
    void *stream_) const {

    auto stream = reinterpret_cast<cudaStream_t>(stream_);
    if (_info.dtype == INFINI_DTYPE_F16) {
        return launch_typed<half>(workspace, workspace_size, _info, out, hidden, topk_indices, topk_weights, gate_weights, up_weights, down_weights, stream, false);
    }
    if (_info.dtype == INFINI_DTYPE_BF16) {
        return launch_typed<__nv_bfloat16>(workspace, workspace_size, _info, out, hidden, topk_indices, topk_weights, gate_weights, up_weights, down_weights, stream, false);
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
    void *stream_) const {

    auto stream = reinterpret_cast<cudaStream_t>(stream_);
    auto gate_weights = reinterpret_cast<const void *const *>(gate_weight_ptrs);
    auto up_weights = reinterpret_cast<const void *const *>(up_weight_ptrs);
    auto down_weights = reinterpret_cast<const void *const *>(down_weight_ptrs);
    if (_info.dtype == INFINI_DTYPE_F16) {
        return launch_typed<half>(workspace, workspace_size, _info, out, hidden, topk_indices, topk_weights, gate_weights, up_weights, down_weights, stream, true);
    }
    if (_info.dtype == INFINI_DTYPE_BF16) {
        return launch_typed<__nv_bfloat16>(workspace, workspace_size, _info, out, hidden, topk_indices, topk_weights, gate_weights, up_weights, down_weights, stream, true);
    }
    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}

} // namespace op::deepseek_moe::nvidia
