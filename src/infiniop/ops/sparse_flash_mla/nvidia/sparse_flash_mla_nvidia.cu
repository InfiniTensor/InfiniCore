#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../tensor.h"
#include "infiniop/ops/sparse_flash_mla.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {
constexpr size_t THREADS = 256;
constexpr float NEG_INF = -3.402823466e+38F;

template <typename T>
__device__ float toFloat(T value);

template <>
__device__ float toFloat(half value) {
    return __half2float(value);
}

template <>
__device__ float toFloat(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T>
__device__ T fromFloat(float value);

template <>
__device__ half fromFloat(float value) {
    return __float2half_rn(value);
}

template <>
__device__ __nv_bfloat16 fromFloat(float value) {
    return __float2bfloat16(value);
}

template <typename T>
INFINIOP_CUDA_KERNEL sparseFlashMlaKernel(
    T *output,
    const T *query,
    const T *kv_cache,
    const int32_t *indices,
    const int32_t *topk_lens,
    size_t tokens,
    size_t heads,
    size_t query_dim,
    size_t value_dim,
    size_t topk,
    size_t cache_slots,
    float scale) {
    const size_t token_head = blockIdx.x;
    if (token_head >= tokens * heads) {
        return;
    }
    const size_t token = token_head / heads;
    const size_t query_offset = token_head * query_dim;
    const size_t indices_offset = token * topk;
    int32_t valid = topk_lens[token];
    valid = valid < 0 ? 0 : valid;
    valid = valid > static_cast<int32_t>(topk) ? static_cast<int32_t>(topk) : valid;

    extern __shared__ float shared[];
    float *scores = shared;
    float *scratch = scores + topk;
    for (int32_t k = 0; k < valid; ++k) {
        const int32_t cache_index = indices[indices_offset + k];
        float partial = 0.0f;
        if (cache_index >= 0 && static_cast<size_t>(cache_index) < cache_slots) {
            const size_t cache_offset = static_cast<size_t>(cache_index) * query_dim;
            for (size_t d = threadIdx.x; d < query_dim; d += blockDim.x) {
                partial += toFloat(query[query_offset + d])
                         * toFloat(kv_cache[cache_offset + d]);
            }
        }
        scratch[threadIdx.x] = partial;
        __syncthreads();
        for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                scratch[threadIdx.x] += scratch[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            scores[k] = cache_index >= 0 && static_cast<size_t>(cache_index) < cache_slots
                          ? scratch[0] * scale
                          : NEG_INF;
        }
        __syncthreads();
    }

    float local_max = NEG_INF;
    for (int32_t k = threadIdx.x; k < valid; k += blockDim.x) {
        local_max = fmaxf(local_max, scores[k]);
    }
    scratch[threadIdx.x] = local_max;
    __syncthreads();
    for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float max_score = scratch[0];

    float local_sum = 0.0f;
    if (max_score != NEG_INF) {
        for (int32_t k = threadIdx.x; k < valid; k += blockDim.x) {
            const float weight = expf(scores[k] - max_score);
            scores[k] = weight;
            local_sum += weight;
        }
    }
    scratch[threadIdx.x] = local_sum;
    __syncthreads();
    for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float denominator = scratch[0];

    for (size_t d = threadIdx.x; d < value_dim; d += blockDim.x) {
        float value = 0.0f;
        if (denominator > 0.0f) {
            for (int32_t k = 0; k < valid; ++k) {
                const int32_t cache_index = indices[indices_offset + k];
                if (cache_index >= 0 && static_cast<size_t>(cache_index) < cache_slots) {
                    value += scores[k] * toFloat(kv_cache[static_cast<size_t>(cache_index) * query_dim + d]);
                }
            }
            value /= denominator;
        }
        output[token_head * value_dim + d] = fromFloat<T>(value);
    }
}
} // namespace

__INFINI_C infiniStatus_t infiniopSparseFlashMla(
    infiniopHandle_t handle,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t query_desc,
    infiniopTensorDescriptor_t kv_cache_desc,
    infiniopTensorDescriptor_t indices_desc,
    infiniopTensorDescriptor_t topk_lens_desc,
    void *output,
    const void *query,
    const void *kv_cache,
    const void *indices,
    const void *topk_lens,
    float scale,
    void *stream) {
    (void)handle;
    const auto output_shape = output_desc->shape();
    const auto query_shape = query_desc->shape();
    const auto cache_shape = kv_cache_desc->shape();
    const auto indices_shape = indices_desc->shape();
    const auto lens_shape = topk_lens_desc->shape();
    CHECK_OR_RETURN(output_shape.size() == 3 && query_shape.size() == 3
                        && cache_shape.size() == 3 && indices_shape.size() == 3
                        && lens_shape.size() == 1,
                    INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(output_shape[0] == query_shape[0]
                        && output_shape[1] == query_shape[1]
                        && output_shape[2] <= query_shape[2]
                        && cache_shape[1] == 1
                        && cache_shape[2] == query_shape[2]
                        && indices_shape[0] == query_shape[0]
                        && indices_shape[1] == 1
                        && lens_shape[0] == query_shape[0],
                    INFINI_STATUS_BAD_TENSOR_SHAPE);
    const auto dtype = query_desc->dtype();
    CHECK_OR_RETURN((dtype == INFINI_DTYPE_F16 || dtype == INFINI_DTYPE_BF16)
                        && output_desc->dtype() == dtype
                        && kv_cache_desc->dtype() == dtype
                        && indices_desc->dtype() == INFINI_DTYPE_I32
                        && topk_lens_desc->dtype() == INFINI_DTYPE_I32,
                    INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(output_desc->isContiguous() && query_desc->isContiguous()
                        && kv_cache_desc->isContiguous() && indices_desc->isContiguous()
                        && topk_lens_desc->isContiguous(),
                    INFINI_STATUS_BAD_TENSOR_STRIDES);

    const size_t tokens = query_shape[0];
    const size_t heads = query_shape[1];
    const size_t query_dim = query_shape[2];
    const size_t value_dim = output_shape[2];
    const size_t topk = indices_shape[2];
    const size_t cache_slots = cache_shape[0];
    const size_t shared_bytes = (topk + THREADS) * sizeof(float);
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
#define LAUNCH(T)                                                                    \
    sparseFlashMlaKernel<T><<<tokens * heads, THREADS, shared_bytes, cuda_stream>>>( \
        static_cast<T *>(output), static_cast<const T *>(query),                     \
        static_cast<const T *>(kv_cache), static_cast<const int32_t *>(indices),     \
        static_cast<const int32_t *>(topk_lens), tokens, heads, query_dim,           \
        value_dim, topk, cache_slots, scale)
    if (dtype == INFINI_DTYPE_F16) {
        LAUNCH(half);
    } else {
        LAUNCH(__nv_bfloat16);
    }
#undef LAUNCH
    return cudaGetLastError() == cudaSuccess ? INFINI_STATUS_SUCCESS
                                             : INFINI_STATUS_INTERNAL_ERROR;
}
