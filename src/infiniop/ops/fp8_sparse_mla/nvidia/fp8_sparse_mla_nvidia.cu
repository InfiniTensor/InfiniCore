#include "../../../../utils.h"
#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../tensor.h"
#include "fp8_sparse_mla_nvidia.cuh"

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

namespace {
constexpr size_t THREADS = 256;
constexpr size_t GROUP_SIZE = 64;
constexpr size_t LANES_PER_KEY = 4;
constexpr size_t LATENT_DIM = 512;
constexpr size_t ROPE_DIM = 64;
constexpr size_t HEAD_DIM = LATENT_DIM + ROPE_DIM;
constexpr size_t VALUE_DIM = LATENT_DIM;
constexpr size_t CACHE_STRIDE = LATENT_DIM + 4 * sizeof(float)
                              + ROPE_DIM * sizeof(cuda_bfloat16);
constexpr size_t MAX_GROUPS = 64;

INFINIOP_CUDA_KERNEL fp8SparseMlaGroupKernel(
    float *__restrict__ partial,
    float *__restrict__ group_maxs,
    float *__restrict__ group_sums,
    const cuda_bfloat16 *__restrict__ query,
    const uint8_t *__restrict__ kv_cache,
    const int32_t *__restrict__ indices,
    const int32_t *__restrict__ topk_lens,
    size_t num_heads,
    size_t num_cache_tokens,
    size_t topk,
    size_t groups,
    float softmax_scale) {
    const size_t group = blockIdx.x;
    const size_t head = blockIdx.y;
    const size_t token = blockIdx.z;
    const size_t thread = threadIdx.x;
    const size_t lane = thread % LANES_PER_KEY;
    const size_t key_in_group = thread / LANES_PER_KEY;
    const size_t key_slot = group * GROUP_SIZE + key_in_group;

    extern __shared__ float shared[];
    float *logits = shared;
    float *scratch = shared + GROUP_SIZE;

    const int32_t valid_count_raw = topk_lens[token];
    const size_t valid_count = valid_count_raw > 0
                                 ? min(static_cast<size_t>(valid_count_raw), topk)
                                 : 0;
    int32_t cache_index = -1;
    bool valid = key_slot < valid_count;
    if (valid) {
        cache_index = indices[token * topk + key_slot];
        valid = cache_index >= 0
             && static_cast<size_t>(cache_index) < num_cache_tokens;
    }

    float dot = 0.0f;
    if (valid) {
        const auto *cache_entry = kv_cache
                                + static_cast<size_t>(cache_index) * CACHE_STRIDE;
        const auto *latent = reinterpret_cast<const cuda_fp8_e4m3 *>(cache_entry);
        const auto *scales = reinterpret_cast<const float *>(cache_entry + LATENT_DIM);
        const auto *rope = reinterpret_cast<const cuda_bfloat16 *>(
            cache_entry + LATENT_DIM + 4 * sizeof(float));
        const auto *q = query + (token * num_heads + head) * HEAD_DIM;
        for (size_t column = lane; column < LATENT_DIM; column += LANES_PER_KEY) {
            dot += __bfloat162float(q[column])
                 * static_cast<float>(latent[column])
                 * scales[column / 128];
        }
        for (size_t column = lane; column < ROPE_DIM; column += LANES_PER_KEY) {
            dot += __bfloat162float(q[LATENT_DIM + column])
                 * __bfloat162float(rope[column]);
        }
    }
    dot += __shfl_xor_sync(0xffffffffu, dot, 1, LANES_PER_KEY);
    dot += __shfl_xor_sync(0xffffffffu, dot, 2, LANES_PER_KEY);
    if (lane == 0) {
        logits[key_in_group] = valid ? dot * softmax_scale : -CUDART_INF_F;
    }
    __syncthreads();

    if (thread < GROUP_SIZE) {
        scratch[thread] = logits[thread];
    }
    __syncthreads();
    for (size_t stride = GROUP_SIZE / 2; stride > 0; stride >>= 1) {
        if (thread < stride) {
            scratch[thread] = fmaxf(scratch[thread], scratch[thread + stride]);
        }
        __syncthreads();
    }
    const float group_max = scratch[0];

    if (thread < GROUP_SIZE) {
        const float value = logits[thread];
        const float probability = isfinite(value) && isfinite(group_max)
                                    ? expf(value - group_max)
                                    : 0.0f;
        logits[thread] = probability;
        scratch[thread] = probability;
    }
    __syncthreads();
    for (size_t stride = GROUP_SIZE / 2; stride > 0; stride >>= 1) {
        if (thread < stride) {
            scratch[thread] += scratch[thread + stride];
        }
        __syncthreads();
    }
    const float group_sum = scratch[0];
    const size_t stats_offset = (token * num_heads + head) * groups + group;
    if (thread == 0) {
        group_maxs[stats_offset] = group_max;
        group_sums[stats_offset] = group_sum;
    }

    const size_t partial_base = stats_offset * VALUE_DIM;
    for (size_t value_column = thread; value_column < VALUE_DIM;
         value_column += blockDim.x) {
        float acc = 0.0f;
        if (group_sum > 0.0f) {
            for (size_t key = 0; key < GROUP_SIZE; ++key) {
                const size_t slot = group * GROUP_SIZE + key;
                if (slot >= valid_count) {
                    break;
                }
                const int32_t index = indices[token * topk + slot];
                if (index < 0 || static_cast<size_t>(index) >= num_cache_tokens) {
                    continue;
                }
                const auto *cache_entry = kv_cache
                                        + static_cast<size_t>(index) * CACHE_STRIDE;
                const auto *latent = reinterpret_cast<const cuda_fp8_e4m3 *>(cache_entry);
                const auto *scales = reinterpret_cast<const float *>(cache_entry + LATENT_DIM);
                acc += (logits[key] / group_sum)
                     * static_cast<float>(latent[value_column])
                     * scales[value_column / 128];
            }
        }
        partial[partial_base + value_column] = acc;
    }
}

INFINIOP_CUDA_KERNEL fp8SparseMlaReduceKernel(
    cuda_bfloat16 *__restrict__ output,
    const float *__restrict__ partial,
    const float *__restrict__ group_maxs,
    const float *__restrict__ group_sums,
    size_t num_heads,
    size_t groups) {
    const size_t value_tile = blockIdx.x;
    const size_t head = blockIdx.y;
    const size_t token = blockIdx.z;
    const size_t thread = threadIdx.x;

    extern __shared__ float shared[];
    float *max_values = shared;
    float *weights = shared + MAX_GROUPS;
    float *scratch = shared + 2 * MAX_GROUPS;
    const size_t stats_base = (token * num_heads + head) * groups;

    if (thread < MAX_GROUPS) {
        const float value = thread < groups
                              ? group_maxs[stats_base + thread]
                              : -CUDART_INF_F;
        max_values[thread] = value;
        scratch[thread] = value;
    }
    __syncthreads();
    for (size_t stride = MAX_GROUPS / 2; stride > 0; stride >>= 1) {
        if (thread < stride) {
            scratch[thread] = fmaxf(scratch[thread], scratch[thread + stride]);
        }
        __syncthreads();
    }
    const float final_max = scratch[0];

    if (thread < MAX_GROUPS) {
        const float weight = thread < groups
                                  && isfinite(max_values[thread])
                                  && isfinite(final_max)
                               ? expf(max_values[thread] - final_max)
                                     * group_sums[stats_base + thread]
                               : 0.0f;
        weights[thread] = weight;
        scratch[thread] = weight;
    }
    __syncthreads();
    for (size_t stride = MAX_GROUPS / 2; stride > 0; stride >>= 1) {
        if (thread < stride) {
            scratch[thread] += scratch[thread + stride];
        }
        __syncthreads();
    }
    const float denominator = scratch[0];

    const size_t value_column = value_tile * blockDim.x + thread;
    if (value_column < VALUE_DIM) {
        float acc = 0.0f;
        for (size_t group = 0; group < groups; ++group) {
            const size_t partial_offset = (stats_base + group) * VALUE_DIM;
            acc += partial[partial_offset + value_column] * weights[group];
        }
        const float value = denominator > 0.0f ? acc / denominator : 0.0f;
        output[(token * num_heads + head) * VALUE_DIM + value_column]
            = cuda_bfloat16(value);
    }
}
} // namespace

namespace op::fp8_sparse_mla::nvidia {

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t query_desc,
    infiniopTensorDescriptor_t kv_cache_desc,
    infiniopTensorDescriptor_t indices_desc,
    infiniopTensorDescriptor_t topk_lens_desc,
    float scale) {
    const auto output_shape = output_desc->shape();
    const auto query_shape = query_desc->shape();
    const auto cache_shape = kv_cache_desc->shape();
    const auto indices_shape = indices_desc->shape();
    CHECK_OR_RETURN(output_shape.size() == 3 && query_shape.size() == 3
                        && cache_shape.size() == 3 && indices_shape.size() == 3
                        && topk_lens_desc->shape().size() == 1,
                    INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(output_shape[0] == query_shape[0]
                        && output_shape[1] == query_shape[1]
                        && query_shape[2] == HEAD_DIM
                        && output_shape[2] == VALUE_DIM
                        && cache_shape[1] == 1
                        && cache_shape[2] == CACHE_STRIDE
                        && indices_shape[0] == query_shape[0]
                        && indices_shape[1] == 1
                        && topk_lens_desc->shape()[0] == query_shape[0],
                    INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(query_desc->dtype() == INFINI_DTYPE_BF16
                        && output_desc->dtype() == INFINI_DTYPE_BF16
                        && kv_cache_desc->dtype() == INFINI_DTYPE_U8
                        && indices_desc->dtype() == INFINI_DTYPE_I32
                        && topk_lens_desc->dtype() == INFINI_DTYPE_I32,
                    INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(output_desc->isContiguous() && query_desc->isContiguous()
                        && kv_cache_desc->isContiguous()
                        && indices_desc->isContiguous()
                        && topk_lens_desc->isContiguous(),
                    INFINI_STATUS_BAD_TENSOR_STRIDES);
    const size_t topk = indices_shape[2];
    const size_t groups = (topk + GROUP_SIZE - 1) / GROUP_SIZE;
    CHECK_OR_RETURN(topk > 0 && groups <= MAX_GROUPS,
                    INFINI_STATUS_BAD_TENSOR_SHAPE);
    const size_t stats_elems = query_shape[0] * query_shape[1] * groups;
    const size_t workspace_size = (stats_elems * VALUE_DIM + 2 * stats_elems)
                                * sizeof(float);
    *desc_ptr = new Descriptor(
        query_shape[0], query_shape[1], query_shape[2], output_shape[2],
        cache_shape[0], cache_shape[2], topk, groups, workspace_size, scale,
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *query,
    const void *kv_cache,
    const void *indices,
    const void *topk_lens,
    void *stream) const {
    CHECK_OR_RETURN(workspace != nullptr && workspace_size >= _workspace_size,
                    INFINI_STATUS_INSUFFICIENT_WORKSPACE);
    auto *partial = reinterpret_cast<float *>(workspace);
    const size_t stats_elems = _num_tokens * _num_heads * _groups;
    auto *group_maxs = partial + stats_elems * _value_dim;
    auto *group_sums = group_maxs + stats_elems;
    const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const dim3 group_grid(
        static_cast<unsigned int>(_groups),
        static_cast<unsigned int>(_num_heads),
        static_cast<unsigned int>(_num_tokens));
    fp8SparseMlaGroupKernel<<<group_grid, THREADS, 2 * GROUP_SIZE * sizeof(float), cuda_stream>>>(
        partial,
        group_maxs,
        group_sums,
        reinterpret_cast<const cuda_bfloat16 *>(query),
        reinterpret_cast<const uint8_t *>(kv_cache),
        reinterpret_cast<const int32_t *>(indices),
        reinterpret_cast<const int32_t *>(topk_lens),
        _num_heads,
        _num_cache_tokens,
        _topk,
        _groups,
        _scale);
    if (cudaGetLastError() != cudaSuccess) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    const dim3 reduce_grid(
        static_cast<unsigned int>((_value_dim + THREADS - 1) / THREADS),
        static_cast<unsigned int>(_num_heads),
        static_cast<unsigned int>(_num_tokens));
    fp8SparseMlaReduceKernel<<<reduce_grid, THREADS, 3 * MAX_GROUPS * sizeof(float), cuda_stream>>>(
        reinterpret_cast<cuda_bfloat16 *>(output),
        partial,
        group_maxs,
        group_sums,
        _num_heads,
        _groups);
    return cudaGetLastError() == cudaSuccess
             ? INFINI_STATUS_SUCCESS
             : INFINI_STATUS_INTERNAL_ERROR;
}

} // namespace op::fp8_sparse_mla::nvidia
