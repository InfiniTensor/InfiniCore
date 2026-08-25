#ifndef __UPSAMPLE_BILINEAR_MOORE_H__
#define __UPSAMPLE_BILINEAR_MOORE_H__

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>

namespace op::upsample_bilinear::moore {
__device__ __forceinline__ float get_source_coord(
    float scale,
    int out_index,
    bool align_corners) {

    if (align_corners) {
        return static_cast<float>(out_index) * scale;
    } else {
        return (static_cast<float>(out_index) + 0.5f) * scale - 0.5f;
    }
}

__device__ __forceinline__ int clamp(int val, int min_val, int max_val) {
    return max(min_val, min(val, max_val));
}
template <typename T>
__global__ void upsample_bilinear_kernel(
    T *__restrict__ output,
    const T *__restrict__ input,
    size_t N,
    size_t C,
    size_t H_in,
    size_t W_in,
    size_t H_out,
    size_t W_out,
    ptrdiff_t input_stride_n,
    ptrdiff_t input_stride_c,
    ptrdiff_t input_stride_h,
    ptrdiff_t input_stride_w,
    ptrdiff_t output_stride_n,
    ptrdiff_t output_stride_c,
    ptrdiff_t output_stride_h,
    ptrdiff_t output_stride_w,
    float scale_h,
    float scale_w,
    bool align_corners) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = N * C * H_out * W_out;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < total_elements; i += stride) {
        size_t w_out_idx = i % W_out;
        size_t temp = i / W_out;
        size_t h_out_idx = temp % H_out;
        temp /= H_out;
        size_t c_idx = temp % C;
        size_t n_idx = temp / C;

        float h_real = get_source_coord(scale_h, h_out_idx, align_corners);
        float w_real = get_source_coord(scale_w, w_out_idx, align_corners);

        int h0 = static_cast<int>(floorf(h_real));
        int h1 = h0 + 1;
        int w0 = static_cast<int>(floorf(w_real));
        int w1 = w0 + 1;

        float h1_lambda = h_real - h0;
        float h0_lambda = 1.0f - h1_lambda;
        float w1_lambda = w_real - w0;
        float w0_lambda = 1.0f - w1_lambda;

        h0 = clamp(h0, 0, static_cast<int>(H_in) - 1);
        h1 = clamp(h1, 0, static_cast<int>(H_in) - 1);
        w0 = clamp(w0, 0, static_cast<int>(W_in) - 1);
        w1 = clamp(w1, 0, static_cast<int>(W_in) - 1);

        const ptrdiff_t input_base = n_idx * input_stride_n + c_idx * input_stride_c;
        float val00 = static_cast<float>(input[input_base + h0 * input_stride_h + w0 * input_stride_w]);
        float val01 = static_cast<float>(input[input_base + h0 * input_stride_h + w1 * input_stride_w]);
        float val10 = static_cast<float>(input[input_base + h1 * input_stride_h + w0 * input_stride_w]);
        float val11 = static_cast<float>(input[input_base + h1 * input_stride_h + w1 * input_stride_w]);

        float val = h0_lambda * (w0_lambda * val00 + w1_lambda * val01) + h1_lambda * (w0_lambda * val10 + w1_lambda * val11);

        const ptrdiff_t output_offset = n_idx * output_stride_n + c_idx * output_stride_c
                                      + h_out_idx * output_stride_h + w_out_idx * output_stride_w;
        output[output_offset] = static_cast<T>(val);
    }
}

} // namespace op::upsample_bilinear::moore

#endif // __UPSAMPLE_BILINEAR_MOORE_H__
