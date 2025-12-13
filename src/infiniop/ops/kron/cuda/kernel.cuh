#pragma once
#include <cuda_runtime.h>
#include <type_traits>

namespace op::cuda {

template <typename T>
__global__ void kron_kernel(
    T *output,
    const T *a,
    const T *b,
    size_t total_output,
    size_t ndim,
    size_t *a_shape,
    size_t *b_shape,
    size_t *y_shape) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_output) return;

    // Convert linear index to coordinates
    size_t temp = idx;
    size_t y_coords[8];  // Max 8 dimensions
    for (size_t d = ndim; d-- > 0;) {
        y_coords[d] = temp % y_shape[d];
        temp /= y_shape[d];
    }

    // Compute corresponding a and b coordinates
    size_t a_coords[8];
    size_t b_coords[8];
    for (size_t d = 0; d < ndim; ++d) {
        a_coords[d] = y_coords[d] / b_shape[d];
        b_coords[d] = y_coords[d] % b_shape[d];
    }

    // Convert coordinates to linear indices
    size_t a_idx = 0;
    size_t a_stride = 1;
    for (size_t d = ndim; d-- > 0;) {
        a_idx += a_coords[d] * a_stride;
        a_stride *= a_shape[d];
    }

    size_t b_idx = 0;
    size_t b_stride = 1;
    for (size_t d = ndim; d-- > 0;) {
        b_idx += b_coords[d] * b_stride;
        b_stride *= b_shape[d];
    }

    output[idx] = a[a_idx] * b[b_idx];
}

} // namespace op::cuda
