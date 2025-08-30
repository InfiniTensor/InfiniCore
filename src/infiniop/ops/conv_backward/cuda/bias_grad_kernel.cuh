#ifndef __GRAD_CUDA_H__
#define __GRAD_CUDA_H__

#include <cmath>

// 特化模板：对于 bf16 类型，使用 float 进行累加以保持精度
template <typename T>
__global__ void compute_bias_grad_kernel(const T *grad_output, T *grad_bias,
                                         int batch_size, int channels,
                                         int spatial_size) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) {
        return;
    }

    // 使用 float 进行累加以保持精度
    float sum = 0.0f;
    for (int n = 0; n < batch_size; n++) {
        for (int s = 0; s < spatial_size; s++) {
            int idx = n * channels * spatial_size + c * spatial_size + s;
            sum += static_cast<float>(grad_output[idx]);
        }
    }
    grad_bias[c] = static_cast<T>(sum);
}

#endif // __GRAD_CUDA_H__
