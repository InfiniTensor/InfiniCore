#ifndef SILU_CUDA_H
#define SILU_CUDA_H

namespace op::silu::cuda {
typedef struct SiluOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half2>) {
            // For half2, process each half separately using more stable computation
            half2 ones = __float2half2_rn(1.0f);
            half2 neg_x = __hneg2(x);
            half2 exp_neg_x = h2exp(neg_x);
            half2 one_plus_exp = __hadd2(ones, exp_neg_x);
            half2 sigmoid = __h2div(ones, one_plus_exp);
            return __hmul2(x, sigmoid);
        } else if constexpr (std::is_same_v<T, half>) {
            // Use more numerically stable implementation
            float x_f = __half2float(x);
            float result = x_f / (1.0f + expf(-x_f));
            return __float2half(result);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            // Convert to float for computation to maintain precision
            float x_f = __bfloat162float(x);
            float result = x_f / (1.0f + expf(-x_f));
            return __float2bfloat16(result);
        } else if constexpr (std::is_same_v<T, float>) {
            // Use fast math functions for float
            return x / (1.0f + expf(-x));
        } else if constexpr (std::is_same_v<T, double>) {
            return x / (1.0 + exp(-x));
        } else {
            // Fallback
            return x / (1.0f + expf(-x));
        }
    }
} SiluOp;
} // namespace op::silu::cuda

#endif // SILU_CUDA_H