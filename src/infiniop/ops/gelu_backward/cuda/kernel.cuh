#ifndef __GELU_BACKWARD_CUDA_H__
#define __GELU_BACKWARD_CUDA_H__
// #include <cuda_runtime.h>
// #include <math.h>
#include <cmath>

namespace op::gelu_backward::cuda {
typedef struct GeLUBackwardOp {
public:
    static constexpr size_t num_inputs = 2;
    const float sqrt_2_over_pi = sqrtf(2 / 3.14159f);
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x, const T &grad_y) const {
        float x_f = float(x);
        float theta = sqrt_2_over_pi * (x_f + 0.044715f * x_f * x_f * x_f);
        float tanh_theta = tanhf(theta);
        float sech_theta_sq = 1.0f - tanh_theta * tanh_theta;

        float term1 = 0.5f * (1.0f + tanh_theta);
        float term2 = 0.5f * x_f * sech_theta_sq * sqrt_2_over_pi * (1.0f + 0.134145f * x_f * x_f);
        
        return float(grad_y) * (term1 + term2);
    }
} GeLUBackwardOp;
} // namespace op::gelu_backward::cuda

#endif // __GELU_BACKWARD_KERNEL_H__
