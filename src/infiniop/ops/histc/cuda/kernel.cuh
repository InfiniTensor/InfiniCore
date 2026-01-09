#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>

namespace op::cuda {

template <typename T>
__global__ void histc_kernel(
    float *hist,
    const T *input,
    size_t input_size,
    ptrdiff_t input_stride,
    int64_t bins,
    double min_val,
    double max_val) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    double bin_width = (max_val - min_val) / static_cast<double>(bins);

    for (int i = idx; i < static_cast<int>(input_size); i += stride) {
        double val = static_cast<double>(input[i * input_stride]);

        // Skip values outside range
        if (val < min_val || val > max_val) {
            continue;
        }

        // Calculate bin index
        int64_t bin_idx = static_cast<int64_t>((val - min_val) / bin_width);

        // Handle edge case: max_val should go to last bin
        if (bin_idx >= bins) {
            bin_idx = bins - 1;
        }
        if (bin_idx < 0) {
            bin_idx = 0;
        }

        atomicAdd(&hist[bin_idx], 1.0f);
    }
}

} // namespace op::cuda
