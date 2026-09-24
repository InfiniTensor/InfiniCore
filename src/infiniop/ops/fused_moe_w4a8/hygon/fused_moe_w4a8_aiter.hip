// SPDX-License-Identifier: MIT
// The optimized HIP GEMM kernels are adapted from ROCm/AITER.

#include "../fused_moe_w4a8.h"
#include "aiter/moe_w4a8_opt.h"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <type_traits>

namespace op::fused_moe_w4a8::hygon {
namespace {

struct Workspace {
    int8_t *quantized_input;
    float *input_scale;
    void *activated;
    int8_t *quantized_activated;
    float *activated_scale;
    void *gemm1_output;
    void *gemm2_output;
    int32_t *sorted_token_ids;
    int32_t *expert_ids;
    int32_t *counts;
    int32_t *offsets;
    int32_t *cursors;
    int32_t *num_tokens_post_padded;
};

Workspace splitWorkspace(void *workspace, const FusedMoeW4A8Info &info) {
    auto *base = static_cast<uint8_t *>(workspace);
    Workspace result{};
    result.quantized_input = reinterpret_cast<int8_t *>(base);
    base += info.quantizedInputBytes();
    result.input_scale = reinterpret_cast<float *>(base);
    base += info.inputScaleBytes();
    result.activated = base;
    base += info.activatedBytes();
    result.quantized_activated = reinterpret_cast<int8_t *>(base);
    base += info.quantizedActivatedBytes();
    result.activated_scale = reinterpret_cast<float *>(base);
    base += info.activatedScaleBytes();
    result.gemm1_output = base;
    base += info.aiterGemm1Bytes();
    result.gemm2_output = base;
    base += info.aiterGemm2Bytes();
    result.sorted_token_ids = reinterpret_cast<int32_t *>(base);
    base += info.aiterSortedTokenBytes();
    result.expert_ids = reinterpret_cast<int32_t *>(base);
    base += info.aiterExpertBlockBytes();
    result.counts = reinterpret_cast<int32_t *>(base);
    result.offsets = result.counts + info.num_experts;
    result.cursors = result.offsets + info.num_experts + 1;
    result.num_tokens_post_padded = result.cursors + info.num_experts;
    return result;
}

template <typename T>
__device__ __forceinline__ float loadValue(T value) {
    if constexpr (std::is_same_v<T, half_t>) {
        return __half2float(value);
    } else {
        return __bfloat162float(value);
    }
}

template <typename T>
__device__ __forceinline__ T storeValue(float value) {
    return b32_to_b16<T>(value);
}

__device__ __forceinline__ float activate(
    float gate,
    float up,
    infiniopFusedMoeActivation_t activation) {
    if (activation == INFINIOP_FUSED_MOE_ACT_SITUGLU) {
        constexpr float beta = 4.0f;
        constexpr float linear_beta = 25.0f;
        const float situ_gate = beta * tanhf(gate / beta) / (1.0f + expf(-gate));
        const float bounded_up = linear_beta * tanhf(up / linear_beta);
        return situ_gate * bounded_up;
    }
    return gate / (1.0f + expf(-gate)) * up;
}

__global__ void countRoutes(
    int32_t *counts,
    const int32_t *selected_experts,
    size_t route_count,
    size_t num_experts) {
    const size_t route = blockIdx.x * blockDim.x + threadIdx.x;
    if (route < route_count) {
        const int32_t expert = selected_experts[route];
        if (expert >= 0 && static_cast<size_t>(expert) < num_experts) {
            atomicAdd(counts + expert, 1);
        }
    }
}

__global__ void buildRouteOffsets(
    int32_t *offsets,
    int32_t *expert_ids,
    int32_t *num_tokens_post_padded,
    const int32_t *counts,
    size_t num_experts,
    size_t block_size_m) {
    extern __shared__ int32_t scan[];
    const size_t tid = threadIdx.x;
    const int32_t padded = tid < num_experts
                             ? (counts[tid] + block_size_m - 1)
                                   / block_size_m * block_size_m
                             : 0;
    scan[tid] = padded;
    __syncthreads();

    for (size_t delta = 1; delta < blockDim.x; delta <<= 1) {
        const int32_t addend = tid >= delta ? scan[tid - delta] : 0;
        __syncthreads();
        scan[tid] += addend;
        __syncthreads();
    }

    if (tid < num_experts) {
        const int32_t begin = tid == 0 ? 0 : scan[tid - 1];
        const int32_t end = scan[tid];
        offsets[tid] = begin;
        for (int32_t row = begin; row < end; row += block_size_m) {
            expert_ids[row / block_size_m] = static_cast<int32_t>(tid);
        }
    }
    if (tid == 0) {
        const int32_t total = scan[num_experts - 1];
        offsets[num_experts] = total;
        *num_tokens_post_padded = total;
    }
}

size_t nextPowerOfTwo(size_t value) {
    size_t result = 1;
    while (result < value) {
        result <<= 1;
    }
    return result;
}

__global__ void fillSortedTokenIds(
    int32_t *sorted_token_ids,
    size_t capacity,
    int32_t sentinel) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < capacity) {
        sorted_token_ids[index] = sentinel;
    }
}

__global__ void scatterRoutes(
    int32_t *sorted_token_ids,
    int32_t *cursors,
    const int32_t *offsets,
    const int32_t *selected_experts,
    size_t route_count,
    size_t num_experts) {
    const size_t route = blockIdx.x * blockDim.x + threadIdx.x;
    if (route >= route_count) {
        return;
    }
    const int32_t expert = selected_experts[route];
    if (expert >= 0 && static_cast<size_t>(expert) < num_experts) {
        const int32_t local = atomicAdd(cursors + expert, 1);
        sorted_token_ids[offsets[expert] + local] = static_cast<int32_t>(route);
    }
}

__global__ void buildSmallRouteMetadata(
    int32_t *sorted_token_ids,
    int32_t *expert_ids,
    int32_t *num_tokens_post_padded,
    const int32_t *selected_experts,
    size_t route_count,
    size_t num_experts,
    size_t block_size_m) {
    extern __shared__ int32_t shared[];
    int32_t *experts = shared;
    int32_t *routes = experts + blockDim.x;
    const size_t tid = threadIdx.x;
    const int32_t expert = tid < route_count ? selected_experts[tid] : -1;
    experts[tid] = expert >= 0 && static_cast<size_t>(expert) < num_experts
                     ? expert
                     : INT32_MAX;
    routes[tid] = static_cast<int32_t>(tid);
    __syncthreads();

    for (size_t width = 2; width <= blockDim.x; width <<= 1) {
        for (size_t stride = width >> 1; stride > 0; stride >>= 1) {
            const size_t other = tid ^ stride;
            if (other > tid) {
                const bool ascending = (tid & width) == 0;
                const bool swap = ascending
                                    ? experts[tid] > experts[other]
                                    : experts[tid] < experts[other];
                if (swap) {
                    const int32_t other_expert = experts[other];
                    const int32_t other_route = routes[other];
                    experts[other] = experts[tid];
                    routes[other] = routes[tid];
                    experts[tid] = other_expert;
                    routes[tid] = other_route;
                }
            }
            __syncthreads();
        }
    }

    if (tid != 0) {
        return;
    }
    int32_t output_offset = 0;
    size_t begin = 0;
    while (begin < route_count && experts[begin] != INT32_MAX) {
        size_t end = begin + 1;
        while (end < route_count && experts[end] == experts[begin]) {
            ++end;
        }
        const size_t count = end - begin;
        const size_t padded = (count + block_size_m - 1) / block_size_m
                            * block_size_m;
        for (size_t row = 0; row < padded; ++row) {
            sorted_token_ids[output_offset + row]
                = row < count ? routes[begin + row]
                              : static_cast<int32_t>(route_count);
        }
        for (size_t row = 0; row < padded; row += block_size_m) {
            expert_ids[(output_offset + row) / block_size_m] = experts[begin];
        }
        output_offset += static_cast<int32_t>(padded);
        begin = end;
    }
    *num_tokens_post_padded = output_offset;
}

template <typename T, bool Expanded>
__global__ void quantizeRows(
    int8_t *output,
    float *scales,
    const T *input,
    size_t rows,
    size_t topk,
    size_t cols) {
    const size_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const size_t input_row_index = Expanded ? row / topk : row;
    const T *input_row = input + input_row_index * cols;
    int8_t *output_row = output + row * cols;
    float max_abs = 0.0f;
    for (size_t col = threadIdx.x; col < cols; col += blockDim.x) {
        max_abs = fmaxf(max_abs, fabsf(loadValue(input_row[col])));
    }
    extern __shared__ float scratch[];
    scratch[threadIdx.x] = max_abs;
    __syncthreads();
    for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float scale = fmaxf(scratch[0] / 127.0f, 1.0e-8f);
    if (threadIdx.x == 0) {
        scales[row] = scale;
    }
    const float inv_scale = 1.0f / scale;
    for (size_t col = threadIdx.x; col < cols; col += blockDim.x) {
        int value = static_cast<int>(roundf(loadValue(input_row[col]) * inv_scale));
        value = value > 127 ? 127 : value;
        value = value < -127 ? -127 : value;
        output_row[col] = static_cast<int8_t>(value);
    }
}

template <typename T>
__global__ void activateRoutes(
    T *output,
    const T *gate_up,
    size_t route_count,
    size_t intermediate_size,
    infiniopFusedMoeActivation_t activation_type) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= route_count * intermediate_size) {
        return;
    }
    const size_t route = index / intermediate_size;
    const size_t inner = index % intermediate_size;
    const T *row = gate_up + route * 2 * intermediate_size;
    output[index] = storeValue<T>(activate(
        loadValue(row[inner]),
        loadValue(row[intermediate_size + inner]),
        activation_type));
}

template <typename T>
__global__ void sumRoutes(
    T *output,
    const T *route_output,
    size_t num_tokens,
    size_t topk,
    size_t hidden_size) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_tokens * hidden_size) {
        return;
    }
    const size_t token = index / hidden_size;
    const size_t hidden = index % hidden_size;
    float value = 0.0f;
    for (size_t route = 0; route < topk; ++route) {
        value += loadValue(route_output[(token * topk + route) * hidden_size + hidden]);
    }
    output[index] = storeValue<T>(value);
}

struct KernelModes {
    int gemm1;
    int gemm2;
};

KernelModes selectKernelModes(size_t num_tokens) {
    if (num_tokens <= 128) {
        return {374116, 384116};
    }
    if (num_tokens <= 512) {
        return {374132, 384132};
    }
    if (num_tokens <= 1024) {
        return {374164, 384164};
    }
    if (num_tokens <= 2048) {
        return {374348, 384248};
    }
    return {374364, 41364};
}

template <typename T>
bool dispatchKernel(
    const std::unordered_map<int, KernelFunc_w4a8<T>> &kernels,
    int mode,
    const GemmParams_w4a8<char, T> &params) {
    const auto it = kernels.find(mode);
    if (it == kernels.end()) {
        return false;
    }
    it->second(params);
    return true;
}

template <typename T>
infiniStatus_t launch(
    T *output,
    const T *input,
    const int32_t *selected_experts,
    const float *routing_weights,
    const int8_t *w13_packed,
    const float *w13_scale,
    const int8_t *w2_packed,
    const float *w2_scale,
    void *workspace,
    const FusedMoeW4A8Info &info,
    hipStream_t stream) {
    constexpr size_t threads = 256;
    auto buffers = splitWorkspace(workspace, info);
    const size_t route_count = info.routeCount();
    const size_t sorted_capacity = info.aiterSortedTokenCapacity();

    CHECK_OR_RETURN(
        hipMemsetAsync(buffers.gemm1_output, 0, info.aiterGemm1Bytes(), stream) == hipSuccess,
        INFINI_STATUS_INTERNAL_ERROR);
    CHECK_OR_RETURN(
        hipMemsetAsync(buffers.gemm2_output, 0, info.aiterGemm2Bytes(), stream) == hipSuccess,
        INFINI_STATUS_INTERNAL_ERROR);
    if (route_count <= 256) {
        const size_t sort_threads = nextPowerOfTwo(route_count);
        buildSmallRouteMetadata<<<1, sort_threads,
                                  2 * sort_threads * sizeof(int32_t), stream>>>(
            buffers.sorted_token_ids,
            buffers.expert_ids,
            buffers.num_tokens_post_padded,
            selected_experts,
            route_count,
            info.num_experts,
            info.aiterBlockSizeM());
    } else {
        CHECK_OR_RETURN(
            hipMemsetAsync(buffers.counts, 0, info.aiterRoutingMetadataBytes(), stream)
                == hipSuccess,
            INFINI_STATUS_INTERNAL_ERROR);
        countRoutes<<<(route_count + threads - 1) / threads, threads, 0, stream>>>(
            buffers.counts, selected_experts, route_count, info.num_experts);
        const size_t route_scan_threads = nextPowerOfTwo(info.num_experts);
        CHECK_OR_RETURN(route_scan_threads <= 1024, INFINI_STATUS_BAD_PARAM);
        buildRouteOffsets<<<1, route_scan_threads,
                            route_scan_threads * sizeof(int32_t), stream>>>(
            buffers.offsets, buffers.expert_ids, buffers.num_tokens_post_padded,
            buffers.counts, info.num_experts, info.aiterBlockSizeM());
        fillSortedTokenIds<<<(sorted_capacity + threads - 1) / threads, threads, 0, stream>>>(
            buffers.sorted_token_ids, sorted_capacity, static_cast<int32_t>(route_count));
        scatterRoutes<<<(route_count + threads - 1) / threads, threads, 0, stream>>>(
            buffers.sorted_token_ids, buffers.cursors, buffers.offsets,
            selected_experts, route_count, info.num_experts);
    }

    const bool prefill = info.num_tokens > 512;
    if (prefill) {
        quantizeRows<T, true><<<route_count, threads, threads * sizeof(float), stream>>>(
            buffers.quantized_input, buffers.input_scale, input,
            route_count, info.topk, info.hidden_size);
    } else {
        quantizeRows<T, false><<<info.num_tokens, threads, threads * sizeof(float), stream>>>(
            buffers.quantized_input, buffers.input_scale, input,
            info.num_tokens, 1, info.hidden_size);
    }

    const auto modes = selectKernelModes(info.num_tokens);
    const uint32_t stage1_m = prefill ? route_count : info.num_tokens;
    GemmParams_w4a8<char, T> stage1(
        reinterpret_cast<const char *>(buffers.quantized_input),
        reinterpret_cast<const char *>(w13_packed),
        reinterpret_cast<T *>(buffers.gemm1_output), buffers.input_scale,
        const_cast<float *>(w13_scale), nullptr, buffers.sorted_token_ids,
        buffers.expert_ids, 0, buffers.num_tokens_post_padded,
        stage1_m, 2 * info.intermediate_size, info.hidden_size,
        1, 1, 2 * info.intermediate_size, 1, 1,
        sorted_capacity, info.topk, info.topk, true, stream);
    const bool stage1_launched = prefill
                                   ? dispatchKernel(kernel_maps_gemm1_prefill_w4a8<T>, modes.gemm1, stage1)
                                   : dispatchKernel(kernel_maps_gemm1_decode_w4a8<T>, modes.gemm1, stage1);
    CHECK_OR_RETURN(stage1_launched, INFINI_STATUS_BAD_PARAM);

    activateRoutes<<<(route_count * info.intermediate_size + threads - 1) / threads,
                     threads, 0, stream>>>(
        reinterpret_cast<T *>(buffers.activated),
        reinterpret_cast<const T *>(buffers.gemm1_output), route_count,
        info.intermediate_size, info.activation);
    quantizeRows<T, false><<<route_count, threads, threads * sizeof(float), stream>>>(
        buffers.quantized_activated, buffers.activated_scale,
        reinterpret_cast<const T *>(buffers.activated), route_count,
        1, info.intermediate_size);

    GemmParams_w4a8<char, T> stage2(
        reinterpret_cast<const char *>(buffers.quantized_activated),
        reinterpret_cast<const char *>(w2_packed),
        reinterpret_cast<T *>(buffers.gemm2_output), buffers.activated_scale,
        const_cast<float *>(w2_scale), routing_weights,
        buffers.sorted_token_ids, buffers.expert_ids, 0,
        buffers.num_tokens_post_padded, route_count, info.hidden_size,
        info.intermediate_size, 1, 1, info.hidden_size, 1, 1,
        sorted_capacity, 1, info.topk, true, stream);
    const bool stage2_launched = info.num_tokens <= 512
                                   ? dispatchKernel(kernel_maps_gemm2_decode_w4a8<T>, modes.gemm2, stage2)
                                   : dispatchKernel(kernel_maps_gemm2_prefill_w4a8<T>, modes.gemm2, stage2);
    CHECK_OR_RETURN(stage2_launched, INFINI_STATUS_BAD_PARAM);

    sumRoutes<<<(info.num_tokens * info.hidden_size + threads - 1) / threads,
                threads, 0, stream>>>(
        output, reinterpret_cast<const T *>(buffers.gemm2_output),
        info.num_tokens, info.topk, info.hidden_size);
    CHECK_OR_RETURN(hipGetLastError() == hipSuccess, INFINI_STATUS_INTERNAL_ERROR);
    return INFINI_STATUS_SUCCESS;
}

} // namespace

infiniStatus_t launchAiterFusedMoe(
    void *output,
    const void *input,
    const int32_t *selected_experts,
    const float *routing_weights,
    const int8_t *w13_packed,
    const float *w13_scale,
    const int8_t *w2_packed,
    const float *w2_scale,
    void *workspace,
    const FusedMoeW4A8Info &info,
    void *stream) {
    auto hip_stream = reinterpret_cast<hipStream_t>(stream);
    if (info.dtype == INFINI_DTYPE_F16) {
        return launch(
            static_cast<half_t *>(output), static_cast<const half_t *>(input),
            selected_experts, routing_weights, w13_packed, w13_scale,
            w2_packed, w2_scale, workspace, info, hip_stream);
    }
    if (info.dtype == INFINI_DTYPE_BF16) {
        return launch(
            static_cast<bhalf_t *>(output), static_cast<const bhalf_t *>(input),
            selected_experts, routing_weights, w13_packed, w13_scale,
            w2_packed, w2_scale, workspace, info, hip_stream);
    }
    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}

} // namespace op::fused_moe_w4a8::hygon
