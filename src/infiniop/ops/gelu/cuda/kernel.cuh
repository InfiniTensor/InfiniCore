#ifndef __GELU_CUDA_H__
#define __GELU_CUDA_H__

#include <cmath>

namespace op::gelu::cuda {

typedef struct GeluOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        constexpr float Alpha = 0.7978845608;
        constexpr float Beta = 0.044715;

        if constexpr (std::is_same_v<T, half2>) {
            const half2 alpha = __float2half2_rn(Alpha);
            const half2 beta = __float2half2_rn(Beta);
            const half2 one = __float2half2_rn(1.0f);
            const half2 half_val = __float2half2_rn(0.5f);

            half2 x_cubed = __hmul2(x, __hmul2(x, x)); // x³
            half2 inner = __hfma2(beta, x_cubed, x);   // x + βx³
            half2 tanh_in = __hmul2(alpha, inner);     // α(x + βx³)

            // 向量化tanh近似（避免拆包）
            float2 f_val = __half22float2(tanh_in);
            f_val.x = tanhf(f_val.x);
            f_val.y = tanhf(f_val.y);
            half2 tanh_val = __float22half2_rn(f_val);

            return __hmul2(__hmul2(half_val, x), __hadd2(one, tanh_val)); // 0.5*x*(1+tanh)
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            float x_f = __bfloat162float(x);
            float result = 0.5f * x_f * (1.0f + tanhf(Alpha * (x_f + Beta * x_f * x_f * x_f)));

            return __float2bfloat16(result);
        } else if constexpr (std::is_same_v<T, half>) {
            float x_f = __half2float(x);
            float result = 0.5f * x_f * (1.0f + tanhf(Alpha * (x_f + Beta * x_f * x_f * x_f)));

            return __float2half(result);
        } else if constexpr (std::is_same_v<T, float>) {
            float x_cubed = x * x * x;
            float inner = x + Beta * x_cubed;
            float tanh_val = tanhf(Alpha * inner);

            return 0.5f * x * (1.0f + tanh_val);
        } else {
            double x_cubed = x * x * x;
            double inner = x + static_cast<double>(Beta) * x_cubed;
            double tanh_val = tanh(static_cast<double>(Alpha) * inner);

            return 0.5 * x * (1 + tanh_val);
        }
    }
} GeluOp;

} // namespace op::gelu::cuda

#endif // __GELU_CUDA_H__
