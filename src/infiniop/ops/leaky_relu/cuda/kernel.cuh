#ifndef LEAKY_RELU_CUDA_H
#define LEAKY_RELU_CUDA_H

namespace op::leaky_relu::cuda {
typedef struct LeakyReLUOp {
public:
    static constexpr size_t num_inputs = 1;

    // __host__ __device__ LeakyReLUOp() = default;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &x, const float *negative_slope) const {
        if constexpr (std::is_same_v<T, half2>) {
            // For half2, process each half separately
            half x_low = __low2half(x);
            half x_high = __high2half(x);
            half result_low = x_low >= __float2half(0.0f) ? x_low : __float2half(*negative_slope) * x_low;
            half result_high = x_high >= __float2half(0.0f) ? x_high : __float2half(*negative_slope) * x_high;
            return __halves2half2(result_low, result_high);
        } else if constexpr (std::is_same_v<T, half>) {
            // Use CUDA half operations
            half zero = __float2half(0.0f);
            half neg_slope = __float2half(*negative_slope);
            return x >= zero ? x : neg_slope * x;
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            // Convert to float for computation to maintain precision
            float x_f = __bfloat162float(x);
            float result = x_f >= 0.0f ? x_f : *negative_slope * x_f;
            return __float2bfloat16(result);
        } else if constexpr (std::is_same_v<T, float>) {
            return x >= 0.0f ? x : *negative_slope * x;
        } else if constexpr (std::is_same_v<T, double>) {
            return x >= 0.0 ? x : static_cast<double>(*negative_slope) * x;
        } else {
            // Fallback
            return x >= T(0) ? x : static_cast<T>(*negative_slope) * x;
        }
    }
} LeakyReLUOp;
} // namespace op::leaky_relu::cuda

#endif // LEAKY_RELU_CUDA_H