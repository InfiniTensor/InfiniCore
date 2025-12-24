#pragma once
#include <cuda_runtime.h>
#include <type_traits>

namespace op::cuda {

// Diff kernel: computes n-th order difference along specified dimension
template <typename T>
__global__ void diff_kernel(
    T *output,
    const T *input,
    size_t size_before,
    size_t dim_size,
    size_t size_after,
    int n) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_output = size_before * (dim_size - n) * size_after;

    if (idx >= total_output) return;

    // Calculate position in output tensor
    size_t pos = idx;
    size_t b = pos / ((dim_size - n) * size_after);
    pos %= ((dim_size - n) * size_after);
    size_t i = pos / size_after;
    size_t a = pos % size_after;

    // Compute n-th order difference
    // For n=1: output[i] = input[i+1] - input[i]
    // For n>1: recursively apply
    T result = input[(b * dim_size + (i + n)) * size_after + a];
    T sign = (n % 2 == 0) ? 1 : -1;
    for (int k = 0; k < n; ++k) {
        T coeff = 1.0;
        for (int j = 0; j < k; ++j) {
            coeff *= static_cast<T>(n - j) / static_cast<T>(j + 1);
        }
        if (k % 2 == 1) coeff = -coeff;
        result += coeff * input[(b * dim_size + (i + n - k - 1)) * size_after + a];
    }

    output[idx] = result;
}

} // namespace op::cuda
