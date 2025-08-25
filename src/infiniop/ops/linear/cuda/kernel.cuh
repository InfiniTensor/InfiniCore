#ifndef __LINEAR_CUDA_H__
#define __LINEAR_CUDA_H__

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace op::linear::cuda {

template<typename T>
__global__ void linear_kernel(
    T *y,
    const T *x,
    const T *w,
    const T *b,
    const int *x_shape,
    const int *w_shape,
    const int *x_strides,
    const int *w_strides,
    const int *y_strides,
    int batch_size,
    int in_features,
    int out_features,
    size_t total_output_elements) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_output_elements) return;
    
    // Calculate batch and output feature indices from linear index
    int batch_idx = idx / out_features;
    int out_idx = idx % out_features;
    
    if (batch_idx >= batch_size || out_idx >= out_features) return;
    
    // Perform dot product: y[batch, out] = sum(x[batch, in] * w[out, in]) + b[out]
    // Note: w is stored as [out_features, in_features], so w[out, in] = w[out * in_features + in]
    // Use float accumulation for better precision with half precision types
    float sum = 0.0f;
    for (int in_idx = 0; in_idx < in_features; in_idx++) {
        // Use strides to calculate correct memory offsets
        int x_offset = batch_idx * x_strides[0] + in_idx * x_strides[1];
        int w_offset = out_idx * w_strides[0] + in_idx * w_strides[1];
        sum += static_cast<float>(x[x_offset]) * static_cast<float>(w[w_offset]);
    }
    if (b != nullptr) {
        sum += static_cast<float>(b[out_idx]);
    }
    
    // Use y_strides to calculate output offset
    int y_offset = batch_idx * y_strides[0] + out_idx * y_strides[1];
    y[y_offset] = static_cast<T>(sum);
}

} // namespace op::linear::cuda

#endif // __LINEAR_CUDA_H__