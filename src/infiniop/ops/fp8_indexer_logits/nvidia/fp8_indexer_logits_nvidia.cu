#include "../../../../utils.h"
#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../tensor.h"
#include "fp8_indexer_logits_nvidia.cuh"

#include <cstdint>
#include <cuda_runtime.h>

namespace {
constexpr size_t THREADS = 256;
constexpr size_t LANES_PER_KEY = 4;

INFINIOP_CUDA_KERNEL fp8IndexerLogitsKernel(
    float *__restrict__ logits,
    const cuda_fp8_e4m3 *__restrict__ q_fp8,
    const uint8_t *__restrict__ kv_cache,
    const int32_t *__restrict__ block_tables,
    const float *__restrict__ weights_fp32,
    const int64_t *__restrict__ positions,
    const int32_t *__restrict__ request_ids,
    size_t num_heads,
    size_t head_dim,
    size_t num_cache_blocks,
    size_t block_size,
    size_t cache_stride,
    size_t num_requests,
    size_t max_blocks_per_request,
    size_t max_context_len) {
    const size_t token = blockIdx.x;
    const size_t logical_block = blockIdx.y;
    const size_t lane = threadIdx.x % LANES_PER_KEY;
    const size_t key_in_block = threadIdx.x / LANES_PER_KEY;

    extern __shared__ uint8_t shared_bytes[];
    auto *shared_q = reinterpret_cast<cuda_fp8_e4m3 *>(shared_bytes);
    auto *shared_weights = reinterpret_cast<float *>(
        shared_bytes + num_heads * head_dim);
    const size_t q_base = token * num_heads * head_dim;
    for (size_t i = threadIdx.x; i < num_heads * head_dim; i += blockDim.x) {
        shared_q[i] = q_fp8[q_base + i];
    }
    for (size_t i = threadIdx.x; i < num_heads; i += blockDim.x) {
        shared_weights[i] = weights_fp32[token * num_heads + i];
    }
    __syncthreads();

    const int64_t position = positions[token];
    const int32_t request = request_ids[token];
    const size_t key_position = logical_block * block_size + key_in_block;
    bool valid = key_in_block < block_size
              && key_position < max_context_len
              && position >= 0
              && key_position <= static_cast<size_t>(position)
              && request >= 0
              && static_cast<size_t>(request) < num_requests
              && logical_block < max_blocks_per_request;
    int32_t physical_block = -1;
    if (valid) {
        physical_block = block_tables[static_cast<size_t>(request) * max_blocks_per_request + logical_block];
        valid = physical_block >= 0
             && static_cast<size_t>(physical_block) < num_cache_blocks;
    }

    float acc = 0.0f;
    if (valid) {
        const uint8_t *cache_block = kv_cache
                                   + static_cast<size_t>(physical_block) * block_size * cache_stride;
        const auto *key = reinterpret_cast<const cuda_fp8_e4m3 *>(
            cache_block + key_in_block * head_dim);
        const float key_scale = *reinterpret_cast<const float *>(
            cache_block + block_size * head_dim + key_in_block * sizeof(float));
        for (size_t head = 0; head < num_heads; ++head) {
            float dot = 0.0f;
            const auto *query = shared_q + head * head_dim;
            for (size_t column = lane; column < head_dim; column += LANES_PER_KEY) {
                dot += static_cast<float>(query[column])
                     * static_cast<float>(key[column]);
            }
            dot += __shfl_xor_sync(0xffffffffu, dot, 1, LANES_PER_KEY);
            dot += __shfl_xor_sync(0xffffffffu, dot, 2, LANES_PER_KEY);
            if (lane == 0) {
                acc += fmaxf(dot * key_scale, 0.0f) * shared_weights[head];
            }
        }
    }
    if (lane == 0 && key_position < max_context_len) {
        logits[token * max_context_len + key_position] = valid
                                                           ? acc
                                                           : -CUDART_INF_F;
    }
}
} // namespace

namespace op::fp8_indexer_logits::nvidia {

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t logits_desc,
    infiniopTensorDescriptor_t q_fp8_desc,
    infiniopTensorDescriptor_t kv_cache_desc,
    infiniopTensorDescriptor_t block_tables_desc,
    infiniopTensorDescriptor_t weights_fp32_desc,
    infiniopTensorDescriptor_t positions_desc,
    infiniopTensorDescriptor_t request_ids_desc) {
    const auto logits_shape = logits_desc->shape();
    const auto q_shape = q_fp8_desc->shape();
    const auto cache_shape = kv_cache_desc->shape();
    const auto blocks_shape = block_tables_desc->shape();
    const auto weights_shape = weights_fp32_desc->shape();
    CHECK_OR_RETURN(logits_shape.size() == 2 && q_shape.size() == 3
                        && cache_shape.size() == 3 && blocks_shape.size() == 2
                        && weights_shape.size() == 2
                        && positions_desc->shape().size() == 1
                        && request_ids_desc->shape().size() == 1,
                    INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(logits_shape[0] == q_shape[0]
                        && weights_shape[0] == q_shape[0]
                        && weights_shape[1] == q_shape[1]
                        && positions_desc->shape()[0] == q_shape[0]
                        && request_ids_desc->shape()[0] == q_shape[0]
                        && cache_shape[1] == 64
                        && q_shape[2] == 128
                        && cache_shape[2] == q_shape[2] + sizeof(float),
                    INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(logits_desc->isContiguous() && q_fp8_desc->isContiguous()
                        && kv_cache_desc->isContiguous()
                        && block_tables_desc->isContiguous()
                        && weights_fp32_desc->isContiguous()
                        && positions_desc->isContiguous()
                        && request_ids_desc->isContiguous(),
                    INFINI_STATUS_BAD_TENSOR_STRIDES);
    CHECK_OR_RETURN(logits_desc->dtype() == INFINI_DTYPE_F32
                        && q_fp8_desc->dtype() == INFINI_DTYPE_F8
                        && kv_cache_desc->dtype() == INFINI_DTYPE_U8
                        && block_tables_desc->dtype() == INFINI_DTYPE_I32
                        && weights_fp32_desc->dtype() == INFINI_DTYPE_F32
                        && positions_desc->dtype() == INFINI_DTYPE_I64
                        && request_ids_desc->dtype() == INFINI_DTYPE_I32,
                    INFINI_STATUS_BAD_TENSOR_DTYPE);
    *desc_ptr = new Descriptor(
        q_shape[0], q_shape[1], q_shape[2], cache_shape[0], cache_shape[1],
        cache_shape[2], blocks_shape[0], blocks_shape[1], logits_shape[1],
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *logits,
    const void *q_fp8,
    const void *kv_cache,
    const void *block_tables,
    const void *weights_fp32,
    const void *positions,
    const void *request_ids,
    void *stream) const {
    const dim3 grid(
        static_cast<unsigned int>(_num_tokens),
        static_cast<unsigned int>((_max_context_len + _block_size - 1) / _block_size));
    const size_t smem = _num_heads * _head_dim + _num_heads * sizeof(float);
    fp8IndexerLogitsKernel<<<grid, THREADS, smem, reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<float *>(logits),
        reinterpret_cast<const cuda_fp8_e4m3 *>(q_fp8),
        reinterpret_cast<const uint8_t *>(kv_cache),
        reinterpret_cast<const int32_t *>(block_tables),
        reinterpret_cast<const float *>(weights_fp32),
        reinterpret_cast<const int64_t *>(positions),
        reinterpret_cast<const int32_t *>(request_ids),
        _num_heads, _head_dim, _num_cache_blocks, _block_size, _cache_stride,
        _num_requests, _max_blocks_per_request, _max_context_len);
    return cudaGetLastError() == cudaSuccess
             ? INFINI_STATUS_SUCCESS
             : INFINI_STATUS_INTERNAL_ERROR;
}

} // namespace op::fp8_indexer_logits::nvidia
