#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../tensor.h"
#include "moe_argsort_bincount_nvidia.cuh"

#include <cstdint>
#include <cuda_runtime.h>

namespace {
constexpr size_t THREADS = 256;

INFINIOP_CUDA_KERNEL countExperts(
    int32_t *counts, const int32_t *topk_ids, size_t total, size_t experts) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total) {
        const int32_t expert = topk_ids[index];
        if (expert >= 0 && static_cast<size_t>(expert) < experts) {
            atomicAdd(counts + expert, 1);
        }
    }
}

INFINIOP_CUDA_KERNEL buildOffsets(
    const int32_t *counts, int32_t *offsets, size_t experts) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int32_t offset = 0;
        for (size_t expert = 0; expert < experts; ++expert) {
            offsets[expert] = offset;
            offset += counts[expert];
        }
    }
}

INFINIOP_CUDA_KERNEL scatterPermutation(
    int32_t *sorted_indices,
    int32_t *inv_pos,
    int32_t *cursors,
    const int32_t *offsets,
    const int32_t *topk_ids,
    size_t total,
    size_t experts) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) {
        return;
    }
    const int32_t expert = topk_ids[index];
    if (expert < 0 || static_cast<size_t>(expert) >= experts) {
        return;
    }
    const int32_t position = offsets[expert] + atomicAdd(cursors + expert, 1);
    sorted_indices[position] = static_cast<int32_t>(index);
    inv_pos[index] = position;
}

constexpr size_t FUSED_THREADS = 512;

INFINIOP_CUDA_KERNEL fusedSmallMoeArgsortBincount(
    int32_t *counts,
    int32_t *sorted_indices,
    int32_t *inv_pos,
    const int32_t *topk_ids,
    size_t total,
    size_t experts) {
    __shared__ int32_t prefix[FUSED_THREADS];
    const size_t tid = threadIdx.x;

    int32_t count = 0;
    if (tid < experts) {
        for (size_t index = 0; index < total; ++index) {
            count += topk_ids[index] == static_cast<int32_t>(tid);
        }
        counts[tid] = count;
    }
    prefix[tid] = count;
    __syncthreads();

    for (size_t shift = 1; shift < FUSED_THREADS; shift <<= 1) {
        const int32_t addend = tid >= shift ? prefix[tid - shift] : 0;
        __syncthreads();
        prefix[tid] += addend;
        __syncthreads();
    }

    if (tid >= total) {
        return;
    }
    const int32_t expert = topk_ids[tid];
    if (expert < 0 || static_cast<size_t>(expert) >= experts) {
        return;
    }
    int32_t local_offset = 0;
    for (size_t index = 0; index < tid; ++index) {
        local_offset += topk_ids[index] == expert;
    }
    const int32_t expert_offset = expert == 0 ? 0 : prefix[expert - 1];
    const int32_t position = expert_offset + local_offset;
    sorted_indices[position] = static_cast<int32_t>(tid);
    inv_pos[tid] = position;
}
} // namespace

namespace op::moe_argsort_bincount::nvidia {

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t tokens_per_experts_desc,
    infiniopTensorDescriptor_t sorted_indices_desc,
    infiniopTensorDescriptor_t inv_pos_desc,
    infiniopTensorDescriptor_t topk_ids_desc,
    size_t num_experts) {
    CHECK_OR_RETURN(num_experts > 0 && num_experts <= 512,
                    INFINI_STATUS_BAD_PARAM);
    const size_t total = topk_ids_desc->numel();
    CHECK_OR_RETURN(tokens_per_experts_desc->shape().size() == 1
                        && tokens_per_experts_desc->shape()[0] == num_experts
                        && sorted_indices_desc->shape().size() == 1
                        && sorted_indices_desc->shape()[0] == total
                        && inv_pos_desc->shape().size() == 1
                        && inv_pos_desc->shape()[0] == total,
                    INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(tokens_per_experts_desc->dtype() == INFINI_DTYPE_I32
                        && sorted_indices_desc->dtype() == INFINI_DTYPE_I32
                        && inv_pos_desc->dtype() == INFINI_DTYPE_I32
                        && topk_ids_desc->dtype() == INFINI_DTYPE_I32,
                    INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(tokens_per_experts_desc->isContiguous()
                        && sorted_indices_desc->isContiguous()
                        && inv_pos_desc->isContiguous()
                        && topk_ids_desc->isContiguous(),
                    INFINI_STATUS_BAD_TENSOR_STRIDES);
    *desc_ptr = new Descriptor(
        total, num_experts, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *tokens_per_experts,
    void *sorted_indices,
    void *inv_pos,
    const void *topk_ids,
    void *stream) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    auto counts = static_cast<int32_t *>(tokens_per_experts);
    if (_total <= FUSED_THREADS && _num_experts <= FUSED_THREADS) {
        fusedSmallMoeArgsortBincount<<<1, FUSED_THREADS, 0, cuda_stream>>>(
            counts, static_cast<int32_t *>(sorted_indices),
            static_cast<int32_t *>(inv_pos),
            static_cast<const int32_t *>(topk_ids), _total, _num_experts);
        return cudaGetLastError() == cudaSuccess
                 ? INFINI_STATUS_SUCCESS
                 : INFINI_STATUS_INTERNAL_ERROR;
    }

    auto offsets = static_cast<int32_t *>(workspace);
    auto cursors = offsets + _num_experts;
    cudaMemsetAsync(counts, 0, _num_experts * sizeof(int32_t), cuda_stream);
    cudaMemsetAsync(cursors, 0, _num_experts * sizeof(int32_t), cuda_stream);
    const size_t blocks = (_total + THREADS - 1) / THREADS;
    countExperts<<<blocks, THREADS, 0, cuda_stream>>>(
        counts, static_cast<const int32_t *>(topk_ids), _total, _num_experts);
    buildOffsets<<<1, 1, 0, cuda_stream>>>(counts, offsets, _num_experts);
    scatterPermutation<<<blocks, THREADS, 0, cuda_stream>>>(
        static_cast<int32_t *>(sorted_indices), static_cast<int32_t *>(inv_pos),
        cursors, offsets, static_cast<const int32_t *>(topk_ids),
        _total, _num_experts);
    return cudaGetLastError() == cudaSuccess ? INFINI_STATUS_SUCCESS
                                             : INFINI_STATUS_INTERNAL_ERROR;
}

} // namespace op::moe_argsort_bincount::nvidia
