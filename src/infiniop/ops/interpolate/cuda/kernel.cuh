#pragma once
#include <cuda_runtime.h>
#include <type_traits>
#include <cmath>

namespace op::cuda {

// Bilinear interpolation kernel for 2D
template <typename T>
__global__ void interpolate_bilinear_2d_kernel(
    T *output,
    const T *input,
    size_t batch,
    size_t channels,
    size_t in_h,
    size_t in_w,
    size_t out_h,
    size_t out_w,
    double scale_h,
    double scale_w,
    int align_corners) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch * channels * out_h * out_w;

    if (idx >= total) return;

    size_t n = idx / (channels * out_h * out_w);
    size_t rem = idx % (channels * out_h * out_w);
    size_t c = rem / (out_h * out_w);
    rem = rem % (out_h * out_w);
    size_t oh = rem / out_w;
    size_t ow = rem % out_w;

    double src_y = align_corners ? oh * scale_h : (oh + 0.5) * scale_h - 0.5;
    double src_x = align_corners ? ow * scale_w : (ow + 0.5) * scale_w - 0.5;

    src_y = fmax(0.0, fmin(src_y, static_cast<double>(in_h - 1)));
    src_x = fmax(0.0, fmin(src_x, static_cast<double>(in_w - 1)));

    size_t y0 = static_cast<size_t>(floor(src_y));
    size_t y1 = (y0 + 1 < in_h) ? y0 + 1 : y0;
    size_t x0 = static_cast<size_t>(floor(src_x));
    size_t x1 = (x0 + 1 < in_w) ? x0 + 1 : x0;

    double dy = src_y - y0;
    double dx = src_x - x0;

    T v00 = input[((n * channels + c) * in_h + y0) * in_w + x0];
    T v01 = input[((n * channels + c) * in_h + y0) * in_w + x1];
    T v10 = input[((n * channels + c) * in_h + y1) * in_w + x0];
    T v11 = input[((n * channels + c) * in_h + y1) * in_w + x1];

    double result = (1 - dy) * (1 - dx) * static_cast<double>(v00) +
                   (1 - dy) * dx * static_cast<double>(v01) +
                   dy * (1 - dx) * static_cast<double>(v10) +
                   dy * dx * static_cast<double>(v11);

    output[((n * channels + c) * out_h + oh) * out_w + ow] = static_cast<T>(result);
}

// Nearest neighbor interpolation kernel
template <typename T>
__global__ void interpolate_nearest_kernel(
    T *output,
    const T *input,
    size_t batch,
    size_t channels,
    size_t *input_dims,
    size_t *output_dims,
    size_t ndim,
    double *scales,
    int align_corners) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch * channels;
    for (size_t d = 0; d < ndim; ++d) {
        total *= output_dims[d];
    }

    if (idx >= total) return;

    // Compute output coordinates
    size_t temp = idx;
    size_t coords[8];
    coords[0] = temp / (channels * output_dims[0] * output_dims[1]);
    temp %= (channels * output_dims[0] * output_dims[1]);
    coords[1] = temp / (output_dims[0] * output_dims[1]);
    temp %= (output_dims[0] * output_dims[1]);
    for (size_t d = 0; d < ndim; ++d) {
        coords[d + 2] = temp / (output_dims[ndim - 1 - d]);
        temp %= (output_dims[ndim - 1 - d]);
    }

    // Compute input coordinates
    size_t in_coords[8];
    in_coords[0] = coords[0];
    in_coords[1] = coords[1];
    for (size_t d = 0; d < ndim; ++d) {
        double src = align_corners ? coords[d + 2] * scales[d] : (coords[d + 2] + 0.5) * scales[d] - 0.5;
        in_coords[d + 2] = static_cast<size_t>(round(src));
        if (in_coords[d + 2] >= input_dims[d]) in_coords[d + 2] = input_dims[d] - 1;
    }

    // Compute input index
    size_t in_idx = in_coords[0] * channels * input_dims[0] * input_dims[1] +
                   in_coords[1] * input_dims[0] * input_dims[1] +
                   in_coords[2] * input_dims[1] + in_coords[3];

    output[idx] = input[in_idx];
}

} // namespace op::cuda
