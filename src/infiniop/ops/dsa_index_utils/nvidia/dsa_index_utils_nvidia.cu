#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../tensor.h"
#include "infiniop/ops/dsa_index_utils.h"

#include <cuda_runtime.h>

namespace {
constexpr size_t THREADS = 256;

INFINIOP_CUDA_KERNEL mapDecodeRequestBlockIndicesKernel(
    int32_t *output,
    const int32_t *request_ids,
    const int32_t *block_table,
    const int32_t *token_indices,
    size_t rows,
    size_t indices_per_row,
    size_t requests,
    size_t blocks_per_request,
    int64_t block_size) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * indices_per_row) {
        return;
    }
    const size_t row = index / indices_per_row;
    const int32_t token_index = token_indices[index];
    const int32_t request = request_ids[row];
    if (token_index < 0 || request < 0
        || static_cast<size_t>(request) >= requests) {
        output[index] = -1;
        return;
    }
    const size_t logical_block = static_cast<size_t>(token_index / block_size);
    if (logical_block >= blocks_per_request) {
        output[index] = -1;
        return;
    }
    const int32_t physical_block = block_table[static_cast<size_t>(request) * blocks_per_request + logical_block];
    output[index] = physical_block < 0
                      ? -1
                      : physical_block * static_cast<int32_t>(block_size)
                            + token_index % static_cast<int32_t>(block_size);
}

INFINIOP_CUDA_KERNEL topkIndicesContextLensKernel(
    int32_t *topk_lens,
    const int32_t *indices,
    size_t rows,
    size_t indices_per_row) {
    const size_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    __shared__ int32_t counts[THREADS];
    int32_t valid = 0;
    for (size_t column = threadIdx.x; column < indices_per_row;
         column += blockDim.x) {
        valid += indices[row * indices_per_row + column] >= 0 ? 1 : 0;
    }
    counts[threadIdx.x] = valid;
    __syncthreads();

    for (size_t offset = THREADS / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            counts[threadIdx.x] += counts[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        topk_lens[row] = counts[0];
    }
}
} // namespace

__INFINI_C infiniStatus_t infiniopMapDecodeRequestBlockIndices(
    infiniopHandle_t handle,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t request_ids_desc,
    infiniopTensorDescriptor_t block_table_desc,
    infiniopTensorDescriptor_t token_indices_desc,
    void *output,
    const void *request_ids,
    const void *block_table,
    const void *token_indices,
    int64_t block_size,
    void *stream) {
    (void)handle;
    const auto output_shape = output_desc->shape();
    const auto request_shape = request_ids_desc->shape();
    const auto block_shape = block_table_desc->shape();
    CHECK_OR_RETURN(output_shape.size() >= 2
                        && output_shape == token_indices_desc->shape()
                        && request_shape.size() == 1
                        && request_shape[0] == output_shape[0]
                        && block_shape.size() == 2
                        && block_size > 0,
                    INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(output_desc->dtype() == INFINI_DTYPE_I32
                        && request_ids_desc->dtype() == INFINI_DTYPE_I32
                        && block_table_desc->dtype() == INFINI_DTYPE_I32
                        && token_indices_desc->dtype() == INFINI_DTYPE_I32,
                    INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(output_desc->isContiguous() && request_ids_desc->isContiguous()
                        && block_table_desc->isContiguous()
                        && token_indices_desc->isContiguous(),
                    INFINI_STATUS_BAD_TENSOR_STRIDES);
    const size_t rows = output_shape[0];
    const size_t indices_per_row = output_desc->numel() / rows;
    const size_t total = rows * indices_per_row;
    const size_t blocks = (total + THREADS - 1) / THREADS;
    mapDecodeRequestBlockIndicesKernel<<<
        blocks, THREADS, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        static_cast<int32_t *>(output), static_cast<const int32_t *>(request_ids),
        static_cast<const int32_t *>(block_table),
        static_cast<const int32_t *>(token_indices),
        rows, indices_per_row, block_shape[0], block_shape[1], block_size);
    return cudaGetLastError() == cudaSuccess ? INFINI_STATUS_SUCCESS
                                             : INFINI_STATUS_INTERNAL_ERROR;
}

__INFINI_C infiniStatus_t infiniopTopkIndicesContextLens(
    infiniopHandle_t handle,
    infiniopTensorDescriptor_t topk_lens_desc,
    infiniopTensorDescriptor_t indices_desc,
    void *topk_lens,
    const void *indices,
    void *stream) {
    (void)handle;
    const auto lens_shape = topk_lens_desc->shape();
    const auto indices_shape = indices_desc->shape();
    CHECK_OR_RETURN(lens_shape.size() == 1 && indices_shape.size() >= 2
                        && lens_shape[0] == indices_shape[0],
                    INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(topk_lens_desc->dtype() == INFINI_DTYPE_I32
                        && indices_desc->dtype() == INFINI_DTYPE_I32,
                    INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(topk_lens_desc->isContiguous() && indices_desc->isContiguous(),
                    INFINI_STATUS_BAD_TENSOR_STRIDES);
    const size_t rows = indices_shape[0];
    const size_t indices_per_row = indices_desc->numel() / rows;
    topkIndicesContextLensKernel<<<
        rows, THREADS, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        static_cast<int32_t *>(topk_lens), static_cast<const int32_t *>(indices),
        rows, indices_per_row);
    return cudaGetLastError() == cudaSuccess ? INFINI_STATUS_SUCCESS
                                             : INFINI_STATUS_INTERNAL_ERROR;
}
