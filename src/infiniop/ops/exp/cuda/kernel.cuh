#ifndef __EXP_CUDA_H__
#define __EXP_CUDA_H__

namespace op::exp::cuda {
typedef struct ExpOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half2>) {
            return h2exp(x);
        } else if constexpr (std::is_same_v<T, half>) {
            return hexp(x);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            // 使用double作为中间计算类型以提高精度
            double x_double = static_cast<double>(__bfloat162float(x));
            double result = ::exp(x_double);
            return __float2bfloat16(static_cast<float>(result));
        } else if constexpr (std::is_same_v<T, float>) {
            return expf(x);
        } else {
            return ::exp(x);
        }
    }
} ExpOp;
} // namespace op::exp::cuda

#endif // __EXP_CUDA_H__