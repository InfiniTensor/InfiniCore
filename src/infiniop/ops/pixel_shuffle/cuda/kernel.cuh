#pragma once
#include <cuda_runtime.h>
#include <type_traits>

namespace op::cuda {

template <typename T>
__global__ void pixel_shuffle_kernel(
    T *output,
    const T *input,
    size_t batch,
    size_t out_channels,
    size_t height,
    size_t width,
    int r) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch * out_channels * height * width;

    if (idx >= total) return;

    size_t n = idx / (out_channels * height * width);
    size_t rem = idx % (out_channels * height * width);
    size_t c = rem / (height * width);
    rem = rem % (height * width);
    size_t oh = rem / width;
    size_t ow = rem % width;

    // Calculate input indices
    size_t w = ow / r;
    size_t h = oh / r;
    size_t i = oh % r;
    size_t j = ow % r;
    size_t in_c = c * r * r + i * r + j;

    size_t in_idx = ((n * (out_channels * r * r) + in_c) * (height / r) + h) * (width / r) + w;
    size_t out_idx = ((n * out_channels + c) * height + oh) * width + ow;

    output[out_idx] = input[in_idx];
}

} // namespace op::cuda
