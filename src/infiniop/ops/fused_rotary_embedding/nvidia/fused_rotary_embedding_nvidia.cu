#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../tensor.h"
#include "infiniop/ops/fused_rotary_embedding.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {
constexpr size_t THREADS = 256;

template <typename T>
__device__ float to_float(T value);

template <>
__device__ float to_float(half value) {
    return __half2float(value);
}

template <>
__device__ float to_float(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T>
__device__ T from_float(float value);

template <>
__device__ half from_float(float value) {
    return __float2half_rn(value);
}

template <>
__device__ __nv_bfloat16 from_float(float value) {
    return __float2bfloat16(value);
}

template <typename T>
INFINIOP_CUDA_KERNEL fusedRotaryEmbeddingKernel(
    T *query,
    T *key,
    const int64_t *positions,
    const T *cos_sin_cache,
    size_t tokens,
    size_t query_heads,
    size_t key_heads,
    size_t head_size,
    bool is_neox) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t heads = query_heads + key_heads;
    const size_t half = head_size / 2;
    const size_t total = tokens * heads * half;
    if (index >= total) {
        return;
    }

    const size_t rotary_index = index % half;
    const size_t token_head = index / half;
    const size_t head = token_head % heads;
    const size_t token = token_head / heads;
    const size_t first_dim = is_neox ? rotary_index : rotary_index * 2;
    const size_t second_dim = is_neox ? rotary_index + half : rotary_index * 2 + 1;
    const size_t local_head = head < query_heads ? head : head - query_heads;
    T *tensor = head < query_heads ? query : key;
    const size_t tensor_heads = head < query_heads ? query_heads : key_heads;
    const size_t offset = (token * tensor_heads + local_head) * head_size;
    const size_t cache_offset = static_cast<size_t>(positions[token]) * head_size;
    const float cosine = to_float(cos_sin_cache[cache_offset + rotary_index]);
    const float sine = to_float(cos_sin_cache[cache_offset + half + rotary_index]);
    const float first = to_float(tensor[offset + first_dim]);
    const float second = to_float(tensor[offset + second_dim]);
    tensor[offset + first_dim] = from_float<T>(first * cosine - second * sine);
    tensor[offset + second_dim] = from_float<T>(second * cosine + first * sine);
}
} // namespace

__INFINI_C infiniStatus_t infiniopFusedRotaryEmbedding(
    infiniopHandle_t handle,
    infiniopTensorDescriptor_t query_desc,
    infiniopTensorDescriptor_t key_desc,
    infiniopTensorDescriptor_t positions_desc,
    infiniopTensorDescriptor_t cos_sin_cache_desc,
    void *query,
    void *key,
    const void *positions,
    const void *cos_sin_cache,
    int64_t head_size,
    bool is_neox,
    void *stream) {
    (void)handle;
    const auto query_shape = query_desc->shape();
    const auto key_shape = key_desc->shape();
    const auto positions_shape = positions_desc->shape();
    const auto cache_shape = cos_sin_cache_desc->shape();
    CHECK_OR_RETURN(query_shape.size() == 3 && key_shape.size() == 3
                        && positions_shape.size() == 1 && cache_shape.size() == 2,
                    INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(head_size > 0 && (head_size % 2) == 0
                        && query_shape[0] == key_shape[0]
                        && query_shape[0] == positions_shape[0]
                        && query_shape[2] == static_cast<size_t>(head_size)
                        && key_shape[2] == static_cast<size_t>(head_size)
                        && cache_shape[1] == static_cast<size_t>(head_size),
                    INFINI_STATUS_BAD_TENSOR_SHAPE);
    const auto dtype = query_desc->dtype();
    CHECK_OR_RETURN((dtype == INFINI_DTYPE_F16 || dtype == INFINI_DTYPE_BF16)
                        && key_desc->dtype() == dtype
                        && cos_sin_cache_desc->dtype() == dtype
                        && positions_desc->dtype() == INFINI_DTYPE_I64,
                    INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(query_desc->isContiguous() && key_desc->isContiguous()
                        && positions_desc->isContiguous()
                        && cos_sin_cache_desc->isContiguous(),
                    INFINI_STATUS_BAD_TENSOR_STRIDES);

    const size_t tokens = query_shape[0];
    const size_t query_heads = query_shape[1];
    const size_t key_heads = key_shape[1];
    const size_t total = tokens * (query_heads + key_heads)
                       * static_cast<size_t>(head_size) / 2;
    const size_t blocks = (total + THREADS - 1) / THREADS;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (dtype == INFINI_DTYPE_F16) {
        fusedRotaryEmbeddingKernel<<<blocks, THREADS, 0, cuda_stream>>>(
            static_cast<half *>(query), static_cast<half *>(key),
            static_cast<const int64_t *>(positions), static_cast<const half *>(cos_sin_cache),
            tokens, query_heads, key_heads, head_size, is_neox);
    } else {
        fusedRotaryEmbeddingKernel<<<blocks, THREADS, 0, cuda_stream>>>(
            static_cast<__nv_bfloat16 *>(query), static_cast<__nv_bfloat16 *>(key),
            static_cast<const int64_t *>(positions), static_cast<const __nv_bfloat16 *>(cos_sin_cache),
            tokens, query_heads, key_heads, head_size, is_neox);
    }
    return cudaGetLastError() == cudaSuccess ? INFINI_STATUS_SUCCESS
                                             : INFINI_STATUS_INTERNAL_ERROR;
}
