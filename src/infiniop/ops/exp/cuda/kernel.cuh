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
            return __float2bfloat16(expf(__bfloat162float(x)));
        } else if constexpr (std::is_same_v<T, float>) {
            return expf(x);
        } else {
            return ::exp(x);
        }
    }
} ExpOp;
} // namespace op::exp::cuda

#endif // __EXP_CUDA_H__