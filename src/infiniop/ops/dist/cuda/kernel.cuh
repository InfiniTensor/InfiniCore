#pragma once
#include "../../../reduce/cuda/reduce.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <type_traits>
#include <cmath>

namespace op::cuda {

// Dist kernel: computes p-norm distance between two tensors
template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
__global__ void dist_kernel(
    Tcompute *result,
    const Tdata *x1,
    const Tdata *x2,
    size_t n,
    double p,
    ptrdiff_t x1_stride,
    ptrdiff_t x2_stride) {

    Tcompute sum = 0;

    // Each thread computes partial distance
    for (size_t i = threadIdx.x; i < n; i += BLOCK_SIZE) {
        Tcompute diff = Tcompute(x1[i * x1_stride]) - Tcompute(x2[i * x2_stride]);
        Tcompute abs_diff = fabs(diff);

        if (p == 0.0) {
            if (abs_diff > 1e-10) {
                sum += 1.0;
            }
        } else if (isinf(p)) {
            sum = fmax(sum, abs_diff);
        } else {
            sum += pow(abs_diff, p);
        }
    }

    // Use CUB block-level reduction
    using BlockReduce = cub::BlockReduce<Tcompute, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    Tcompute block_sum = BlockReduce(temp_storage).Sum(sum);

    // Write result (only thread 0, since we only launch 1 block)
    if (threadIdx.x == 0) {
        if (p == 0.0 || isinf(p)) {
            *result = block_sum;
        } else {
            *result = pow(block_sum, 1.0 / p);
        }
    }
}

} // namespace op::cuda
