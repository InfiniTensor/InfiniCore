#ifndef __GELU_BACKWARD_CUDA_H__
#define __GELU_BACKWARD_CUDA_H__

#include <cmath>

namespace op::gelu_backward::cuda {
typedef struct GeluBackwardOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &grad_output, const T &input) const {
        if constexpr (std::is_same_v<T, half>) {
            float fx = __half2float(input);
            float fgrad = __half2float(grad_output);
            float sqrt_2_over_pi = 0.7978845608f; // sqrt(2/pi)
            float tanh_arg = sqrt_2_over_pi * (fx + 0.044715f * fx * fx * fx);
            float tanh_val = tanhf(tanh_arg);
            float sech2_val = 1.0f - tanh_val * tanh_val;
            float dtanh_dx = sqrt_2_over_pi * (1.0f + 3.0f * 0.044715f * fx * fx);
            float dgelu_dx = 0.5f * (1.0f + tanh_val) + 0.5f * fx * sech2_val * dtanh_dx;
            return __float2half(fgrad * dgelu_dx);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            float fx = __bfloat162float(input);
            float fgrad = __bfloat162float(grad_output);
            float sqrt_2_over_pi = 0.7978845608f; // sqrt(2/pi)
            float tanh_arg = sqrt_2_over_pi * (fx + 0.044715f * fx * fx * fx);
            float tanh_val = tanhf(tanh_arg);
            float sech2_val = 1.0f - tanh_val * tanh_val;
            float dtanh_dx = sqrt_2_over_pi * (1.0f + 3.0f * 0.044715f * fx * fx);
            float dgelu_dx = 0.5f * (1.0f + tanh_val) + 0.5f * fx * sech2_val * dtanh_dx;
            return __float2bfloat16(fgrad * dgelu_dx);
        } else if constexpr (std::is_same_v<T, float>) {
            float sqrt_2_over_pi = 0.7978845608f; // sqrt(2/pi)
            float tanh_arg = sqrt_2_over_pi * (input + 0.044715f * input * input * input);
            float tanh_val = tanhf(tanh_arg);
            float sech2_val = 1.0f - tanh_val * tanh_val;
            float dtanh_dx = sqrt_2_over_pi * (1.0f + 3.0f * 0.044715f * input * input);
            float dgelu_dx = 0.5f * (1.0f + tanh_val) + 0.5f * input * sech2_val * dtanh_dx;
            return grad_output * dgelu_dx;
        }
    }
} GeluBackwardOp;
} // namespace op::gelu_backward::cuda

#endif // __GELU_BACKWARD_CUDA_H__