#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../tensor.h"
#include "infiniop/ops/select_decode_topk_block_indices.h"

#include <cub/block/block_radix_sort.cuh>
#include <cuda_runtime.h>

namespace {
constexpr int THREADS = 256;
constexpr int MAX_ITEMS_PER_THREAD = 32;
constexpr size_t MAX_SORT_WIDTH = THREADS * MAX_ITEMS_PER_THREAD;
constexpr float NEG_INF = -3.402823466e+38F;

using MaxBlockRadixSort = cub::BlockRadixSort<float, THREADS, MAX_ITEMS_PER_THREAD, int32_t>;

template <int ITEMS_PER_THREAD>
__device__ __forceinline__ void sortAndStore(
    int32_t *topk_indices,
    const float *logits,
    size_t row,
    size_t columns,
    size_t valid,
    size_t topk,
    typename MaxBlockRadixSort::TempStorage &max_temp_storage) {
    using BlockRadixSort = cub::BlockRadixSort<float, THREADS, ITEMS_PER_THREAD, int32_t>;
    static_assert(
        sizeof(typename BlockRadixSort::TempStorage)
            <= sizeof(typename MaxBlockRadixSort::TempStorage),
        "max radix-sort storage must fit every runtime sort width");
    static_assert(
        alignof(typename BlockRadixSort::TempStorage)
            <= alignof(typename MaxBlockRadixSort::TempStorage),
        "max radix-sort storage must satisfy every runtime alignment");
    auto &temp_storage = *reinterpret_cast<typename BlockRadixSort::TempStorage *>(
        &max_temp_storage);

    float values[ITEMS_PER_THREAD];
    int32_t indices[ITEMS_PER_THREAD];
#pragma unroll
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
        const size_t column = static_cast<size_t>(threadIdx.x) * ITEMS_PER_THREAD + item;
        const bool in_range = column < valid;
        values[item] = in_range ? logits[row * columns + column] : NEG_INF;
        indices[item] = in_range ? static_cast<int32_t>(column) : -1;
    }

    BlockRadixSort(temp_storage).SortDescending(values, indices);

    const size_t selected = valid < topk ? valid : topk;
#pragma unroll
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
        const size_t rank = static_cast<size_t>(threadIdx.x) * ITEMS_PER_THREAD + item;
        if (rank < topk) {
            topk_indices[row * topk + rank] = rank < selected ? indices[item] : -1;
        }
    }
}

INFINIOP_CUDA_KERNEL selectDecodeTopkBlockIndicesDynamicKernel(
    int32_t *topk_indices,
    const float *logits,
    const int32_t *seq_lens,
    size_t rows,
    size_t columns,
    size_t topk) {
    const size_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    int32_t end = seq_lens[row];
    end = end < 0 ? 0 : end;
    end = end > static_cast<int32_t>(columns)
            ? static_cast<int32_t>(columns)
            : end;
    const size_t valid = static_cast<size_t>(end);
    const size_t required = valid > topk ? valid : topk;

    __shared__ typename MaxBlockRadixSort::TempStorage temp_storage;
    if (required > 4096) {
        sortAndStore<32>(
            topk_indices, logits, row, columns, valid, topk, temp_storage);
    } else if (required > 2048) {
        sortAndStore<16>(
            topk_indices, logits, row, columns, valid, topk, temp_storage);
    } else {
        sortAndStore<8>(
            topk_indices, logits, row, columns, valid, topk, temp_storage);
    }
}

template <int ITEMS_PER_THREAD>
INFINIOP_CUDA_KERNEL selectDecodeTopkBlockIndicesStaticKernel(
    int32_t *topk_indices,
    const float *logits,
    const int32_t *seq_lens,
    size_t rows,
    size_t columns,
    size_t topk) {
    const size_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    int32_t end = seq_lens[row];
    end = end < 0 ? 0 : end;
    end = end > static_cast<int32_t>(columns)
            ? static_cast<int32_t>(columns)
            : end;
    const size_t valid = static_cast<size_t>(end);

    float values[ITEMS_PER_THREAD];
    int32_t indices[ITEMS_PER_THREAD];
#pragma unroll
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
        const size_t column = static_cast<size_t>(threadIdx.x) * ITEMS_PER_THREAD + item;
        const bool in_range = column < valid;
        values[item] = in_range ? logits[row * columns + column] : NEG_INF;
        indices[item] = in_range ? static_cast<int32_t>(column) : -1;
    }

    using BlockRadixSort = cub::BlockRadixSort<float, THREADS, ITEMS_PER_THREAD, int32_t>;
    __shared__ typename BlockRadixSort::TempStorage temp_storage;
    BlockRadixSort(temp_storage).SortDescending(values, indices);

    const size_t selected = valid < topk ? valid : topk;
#pragma unroll
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
        const size_t rank = static_cast<size_t>(threadIdx.x) * ITEMS_PER_THREAD + item;
        if (rank < topk) {
            topk_indices[row * topk + rank] = rank < selected ? indices[item] : -1;
        }
    }
}
} // namespace

__INFINI_C infiniStatus_t infiniopSelectDecodeTopkBlockIndices(
    infiniopHandle_t handle,
    infiniopTensorDescriptor_t topk_indices_desc,
    infiniopTensorDescriptor_t logits_desc,
    infiniopTensorDescriptor_t seq_lens_desc,
    void *topk_indices,
    const void *logits,
    const void *seq_lens,
    void *stream) {
    (void)handle;
    const auto out_shape = topk_indices_desc->shape();
    const auto logits_shape = logits_desc->shape();
    CHECK_OR_RETURN(out_shape.size() == 2 && logits_shape.size() == 2
                        && out_shape[0] == logits_shape[0]
                        && out_shape[1] <= logits_shape[1]
                        && seq_lens_desc->shape()
                               == std::vector<size_t>{logits_shape[0]},
                    INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(topk_indices_desc->dtype() == INFINI_DTYPE_I32
                        && logits_desc->dtype() == INFINI_DTYPE_F32
                        && seq_lens_desc->dtype() == INFINI_DTYPE_I32,
                    INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(topk_indices_desc->isContiguous()
                        && logits_desc->isContiguous()
                        && seq_lens_desc->isContiguous(),
                    INFINI_STATUS_BAD_TENSOR_STRIDES);
    CHECK_OR_RETURN(
        logits_shape[1] <= MAX_SORT_WIDTH, INFINI_STATUS_BAD_TENSOR_SHAPE);
    const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (logits_shape[1] == MAX_SORT_WIDTH && out_shape[1] == 2048) {
        selectDecodeTopkBlockIndicesDynamicKernel<<<
            logits_shape[0], THREADS, 0, cuda_stream>>>(
            static_cast<int32_t *>(topk_indices),
            static_cast<const float *>(logits),
            static_cast<const int32_t *>(seq_lens),
            logits_shape[0], logits_shape[1], out_shape[1]);
    } else {
        size_t sort_width = 1;
        while (sort_width < logits_shape[1]) {
            sort_width <<= 1;
        }
#define LAUNCH_BLOCK_RADIX_SORT(ITEMS_PER_THREAD)              \
    selectDecodeTopkBlockIndicesStaticKernel<ITEMS_PER_THREAD> \
        <<<logits_shape[0], THREADS, 0, cuda_stream>>>(        \
            static_cast<int32_t *>(topk_indices),              \
            static_cast<const float *>(logits),                \
            static_cast<const int32_t *>(seq_lens),            \
            logits_shape[0], logits_shape[1], out_shape[1])
        if (sort_width <= 256) {
            LAUNCH_BLOCK_RADIX_SORT(1);
        } else if (sort_width <= 512) {
            LAUNCH_BLOCK_RADIX_SORT(2);
        } else if (sort_width <= 1024) {
            LAUNCH_BLOCK_RADIX_SORT(4);
        } else if (sort_width <= 2048) {
            LAUNCH_BLOCK_RADIX_SORT(8);
        } else if (sort_width <= 4096) {
            LAUNCH_BLOCK_RADIX_SORT(16);
        } else {
            LAUNCH_BLOCK_RADIX_SORT(32);
        }
#undef LAUNCH_BLOCK_RADIX_SORT
    }
    return cudaGetLastError() == cudaSuccess ? INFINI_STATUS_SUCCESS
                                             : INFINI_STATUS_INTERNAL_ERROR;
}
