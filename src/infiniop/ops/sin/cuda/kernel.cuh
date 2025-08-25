#ifndef SIN_CUDA_H
#define SIN_CUDA_H

namespace op::sin::cuda {
typedef struct SinOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half2>) {
            // For half2, process each half separately
            half x_low = __low2half(x);
            half x_high = __high2half(x);
            half sin_low = hsin(x_low);
            half sin_high = hsin(x_high);
            return __halves2half2(sin_low, sin_high);
        } else if constexpr (std::is_same_v<T, half>) {
            // Use CUDA half sin function
            return hsin(x);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            // Convert to float for computation to maintain precision
            float x_f = __bfloat162float(x);
            float result = sinf(x_f);
            return __float2bfloat16(result);
        } else if constexpr (std::is_same_v<T, float>) {
            // Use fast math functions for float
            return sinf(x);
        } else if constexpr (std::is_same_v<T, double>) {
            return ::sin(x);
        } else {
            // Fallback
            return sinf(x);
        }
    }
} SinOp;
} // namespace op::sin::cuda

#endif // SIN_CUDA_H