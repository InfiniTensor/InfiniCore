#ifndef __SIN_CUDA_H__
#define __SIN_CUDA_H__

namespace op::sin::cuda {
typedef struct SinOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half2>) {
            return h2sin(x);
        } else if constexpr (std::is_same_v<T, half>) {
            return hsin(x);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            // 使用double作为中间计算类型以提高精度
            double x_double = static_cast<double>(__bfloat162float(x));
            double result = ::sin(x_double);
            return __float2bfloat16(static_cast<float>(result));
        } else if constexpr (std::is_same_v<T, float>) {
            return sinf(x);
        } else {
            return ::sin(x);
        }
    }
} SinOp;
} // namespace op::sin::cuda

#endif // __SIN_CUDA_H__