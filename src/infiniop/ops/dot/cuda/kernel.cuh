#pragma once
#include "../../../reduce/cuda/reduce.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace op::cuda {

// Dot product kernel: computes dot(a, b) = sum(a * b)
template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
__global__ void dot_kernel(
    Tcompute *result,
    const Tdata *a,
    const Tdata *b,
    size_t n,
    ptrdiff_t a_stride,
    ptrdiff_t b_stride) {

    Tcompute sum = 0;

    // Each thread computes partial dot product
    for (size_t i = threadIdx.x; i < n; i += BLOCK_SIZE) {
        Tcompute a_val = Tcompute(a[i * a_stride]);
        Tcompute b_val = Tcompute(b[i * b_stride]);
        sum += a_val * b_val;
    }

    // Use CUB block-level reduction
    using BlockReduce = cub::BlockReduce<Tcompute, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    Tcompute block_sum = BlockReduce(temp_storage).Sum(sum);

    // Write result (only thread 0, since we only launch 1 block)
    if (threadIdx.x == 0) {
        *result = block_sum;
    }
}

} // namespace op::cuda
