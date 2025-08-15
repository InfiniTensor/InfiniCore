#ifndef __COS_CUDA_H__
#define __COS_CUDA_H__

namespace op::cos::cuda {

typedef struct CosOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half2>) {
            // 对于half2，使用内置函数保持兼容性
            return h2cos(x);
        } else if constexpr (std::is_same_v<T, half>) {
            // 对于half，使用内置函数保持兼容性
            return hcos(x);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            // 对于bfloat16，使用内置函数确保精度
            float x_float = __bfloat162float(x);
            float result = cosf(x_float);
            return __float2bfloat16(result);
        } else if constexpr (std::is_same_v<T, float>) {
            // 对于float，使用内置函数确保精度
            return cosf(x);
        } else {
            // 对于double等其他类型，保持原有实现
            return ::cos(x);
        }
    }
} CosOp;

// 提供一个高精度版本的算子（当需要更高精度时使用）
typedef struct CosOpHighPrecision {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half2>) {
            return h2cos(x);
        } else if constexpr (std::is_same_v<T, half>) {
            return hcos(x);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            // 高精度版本：使用double作为中间计算类型
            double x_double = static_cast<double>(__bfloat162float(x));
            double result = ::cos(x_double);
            return __float2bfloat16(static_cast<float>(result));
        } else if constexpr (std::is_same_v<T, float>) {
            return cosf(x);
        } else {
            return ::cos(x);
        }
    }
} CosOpHighPrecision;

} // namespace op::cos::cuda

#endif // __COS_CUDA_H__