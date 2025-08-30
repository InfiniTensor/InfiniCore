#ifndef __GELU_CUDA_H__
#define __GELU_CUDA_H__

#include <cmath>

namespace op::gelu::cuda {
typedef struct GeluOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half>) {
            float fx = __half2float(x);
            float sqrt_2_over_pi = 0.7978845608f; // sqrt(2/pi)
            float tanh_arg = sqrt_2_over_pi * (fx + 0.044715f * fx * fx * fx);
            float tanh_val = tanhf(tanh_arg);
            float result = 0.5f * fx * (1.0f + tanh_val);
            return __float2half(result);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            float fx = __bfloat162float(x);
            float sqrt_2_over_pi = 0.7978845608f; // sqrt(2/pi)
            float tanh_arg = sqrt_2_over_pi * (fx + 0.044715f * fx * fx * fx);
            float tanh_val = tanhf(tanh_arg);
            float result = 0.5f * fx * (1.0f + tanh_val);
            return __float2bfloat16(result);
        } else if constexpr (std::is_same_v<T, float>) {
            float sqrt_2_over_pi = 0.7978845608f; // sqrt(2/pi)
            float tanh_arg = sqrt_2_over_pi * (x + 0.044715f * x * x * x);
            float tanh_val = tanhf(tanh_arg);
            return 0.5f * x * (1.0f + tanh_val);
        }
    }
} GeluOp;
} // namespace op::gelu::cuda

#endif // __GELU_CUDA_H__