#ifndef COS_CUDA_H
#define COS_CUDA_H

namespace op::cos::cuda {
typedef struct CosOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half2>) {
            // For half2, process each half separately using CUDA intrinsics
            half x_low = __low2half(x);
            half x_high = __high2half(x);

            float x_low_f = __half2float(x_low);
            float x_high_f = __half2float(x_high);

            half cos_low = __float2half(cosf(x_low_f));
            half cos_high = __float2half(cosf(x_high_f));

            return __halves2half2(cos_low, cos_high);
        } else if constexpr (std::is_same_v<T, half>) {
            // Convert to float for computation to maintain precision
            float x_f = __half2float(x);
            float result = cosf(x_f);
            return __float2half(result);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            // Convert to float for computation to maintain precision
            float x_f = __bfloat162float(x);
            float result = cosf(x_f);
            return __float2bfloat16(result);
        } else if constexpr (std::is_same_v<T, float>) {
            // Use fast math functions for float
            return cosf(x);
        } else if constexpr (std::is_same_v<T, double>) {
            return ::cos(x);
        } else {
            // Fallback
            return cosf(x);
        }
    }
} CosOp;
} // namespace op::cos::cuda

#endif // COS_CUDA_H