#ifndef EXP_CUDA_H
#define EXP_CUDA_H

namespace op::exp::cuda {
typedef struct ExpOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a) const {
        if constexpr (std::is_same_v<T, half2>) {
            // For half2, split into two halves, compute exp, then combine
            half2 result;
            result.x = __float2half(expf(__half2float(a.x)));
            result.y = __float2half(expf(__half2float(a.y)));
            return result;
        } else if constexpr (std::is_same_v<T, half>) {
            // Convert half to float, compute exp, convert back
            float fa = __half2float(a);
            float result = expf(fa);
            return __float2half(result);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            // Convert bf16 to float, compute exp, then convert back
            float fa = __bfloat162float(a);
            float result = expf(fa);
            return __float2bfloat16(result);
        } else if constexpr (std::is_same_v<T, float>) {
            return expf(a);
        } else if constexpr (std::is_same_v<T, double>) {
            return ::exp(a);
        } else {
            return ::exp(a);
        }
    }
} ExpOp;
} // namespace op::exp::cuda

#endif // EXP_CUDA_H