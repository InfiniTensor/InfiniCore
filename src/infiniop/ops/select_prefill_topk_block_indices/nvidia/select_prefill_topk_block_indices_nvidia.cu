#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../tensor.h"
#include "infiniop/ops/select_prefill_topk_block_indices.h"

#include <cuda_runtime.h>

namespace {
constexpr size_t THREADS = 256;
constexpr size_t MAX_SORT_WIDTH = 8192;

INFINIOP_CUDA_KERNEL selectPrefillTopkBlockIndicesKernel(
    int32_t *topk_indices,
    const float *logits,
    const int32_t *cu_seqlen_ks,
    const int32_t *cu_seqlen_ke,
    size_t rows,
    size_t columns,
    size_t topk,
    size_t sort_width) {
    const size_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    extern __shared__ unsigned char shared_bytes[];
    auto *values = reinterpret_cast<float *>(shared_bytes);
    auto *indices = reinterpret_cast<int32_t *>(values + sort_width);
    int32_t start = cu_seqlen_ks[row];
    int32_t end = cu_seqlen_ke[row];
    start = start < 0 ? 0 : start;
    end = end < start ? start : end;
    start = start > static_cast<int32_t>(columns)
              ? static_cast<int32_t>(columns)
              : start;
    end = end > static_cast<int32_t>(columns)
            ? static_cast<int32_t>(columns)
            : end;
    const size_t valid = static_cast<size_t>(end - start);

    for (size_t column = threadIdx.x; column < sort_width; column += blockDim.x) {
        const bool in_range = column >= static_cast<size_t>(start)
                           && column < static_cast<size_t>(end);
        values[column] = in_range ? logits[row * columns + column] : -INFINITY;
        indices[column] = in_range ? static_cast<int32_t>(column) : -1;
    }
    for (size_t rank = threadIdx.x; rank < topk; rank += blockDim.x) {
        topk_indices[row * topk + rank] = -1;
    }
    __syncthreads();

    for (size_t width = 2; width <= sort_width; width <<= 1) {
        for (size_t stride = width >> 1; stride > 0; stride >>= 1) {
            for (size_t left = threadIdx.x; left < sort_width; left += blockDim.x) {
                const size_t right = left ^ stride;
                if (right <= left) {
                    continue;
                }
                const bool ascending = (left & width) == 0;
                const bool swap = ascending
                                    ? values[left] > values[right]
                                    : values[left] < values[right];
                if (swap) {
                    const float value = values[left];
                    values[left] = values[right];
                    values[right] = value;
                    const int32_t index = indices[left];
                    indices[left] = indices[right];
                    indices[right] = index;
                }
            }
            __syncthreads();
        }
    }

    const size_t selected = valid < topk ? valid : topk;
    for (size_t rank = threadIdx.x; rank < selected; rank += blockDim.x) {
        topk_indices[row * topk + rank] = indices[sort_width - 1 - rank];
    }
}
} // namespace

__INFINI_C infiniStatus_t infiniopSelectPrefillTopkBlockIndices(
    infiniopHandle_t handle,
    infiniopTensorDescriptor_t topk_indices_desc,
    infiniopTensorDescriptor_t logits_desc,
    infiniopTensorDescriptor_t cu_seqlen_ks_desc,
    infiniopTensorDescriptor_t cu_seqlen_ke_desc,
    void *topk_indices,
    const void *logits,
    const void *cu_seqlen_ks,
    const void *cu_seqlen_ke,
    void *stream) {
    (void)handle;
    const auto out_shape = topk_indices_desc->shape();
    const auto logits_shape = logits_desc->shape();
    CHECK_OR_RETURN(out_shape.size() == 2 && logits_shape.size() == 2
                        && out_shape[0] == logits_shape[0]
                        && cu_seqlen_ks_desc->shape() == std::vector<size_t>{logits_shape[0]}
                        && cu_seqlen_ke_desc->shape() == std::vector<size_t>{logits_shape[0]},
                    INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(topk_indices_desc->dtype() == INFINI_DTYPE_I32
                        && logits_desc->dtype() == INFINI_DTYPE_F32
                        && cu_seqlen_ks_desc->dtype() == INFINI_DTYPE_I32
                        && cu_seqlen_ke_desc->dtype() == INFINI_DTYPE_I32,
                    INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(topk_indices_desc->isContiguous() && logits_desc->isContiguous()
                        && cu_seqlen_ks_desc->isContiguous()
                        && cu_seqlen_ke_desc->isContiguous(),
                    INFINI_STATUS_BAD_TENSOR_STRIDES);
    size_t sort_width = 1;
    while (sort_width < logits_shape[1]) {
        sort_width <<= 1;
    }
    CHECK_OR_RETURN(sort_width <= MAX_SORT_WIDTH, INFINI_STATUS_BAD_TENSOR_SHAPE);
    const size_t shared_memory = sort_width * (sizeof(float) + sizeof(int32_t));
    selectPrefillTopkBlockIndicesKernel<<<
        logits_shape[0], THREADS, shared_memory,
        reinterpret_cast<cudaStream_t>(stream)>>>(
        static_cast<int32_t *>(topk_indices), static_cast<const float *>(logits),
        static_cast<const int32_t *>(cu_seqlen_ks),
        static_cast<const int32_t *>(cu_seqlen_ke),
        logits_shape[0], logits_shape[1], out_shape[1], sort_width);
    return cudaGetLastError() == cudaSuccess ? INFINI_STATUS_SUCCESS
                                             : INFINI_STATUS_INTERNAL_ERROR;
}
