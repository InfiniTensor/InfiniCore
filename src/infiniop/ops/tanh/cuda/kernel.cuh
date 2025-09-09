#ifndef TANH_CUDA_H
#define TANH_CUDA_H

namespace op::tanh::cuda {
typedef struct TanhOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half>) {
            // Use CUDA intrinsic for half precision
            return htanh(x);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            // Convert to float for computation to maintain precision
            float x_f = __bfloat162float(x);
            float result = tanhf(x_f);
            return __float2bfloat16(result);
        } else if constexpr (std::is_same_v<T, float>) {
            // Use fast math functions for float
            return tanhf(x);
            // return static_cast<float>(::tanhf(static_cast<double>(x)));
        } else if constexpr (std::is_same_v<T, double>) {
            return ::tanh(x);
        } else {
            // Fallback
            return tanhf(x);
        }
    }

private:
    // Helper function for half precision tanh (assuming it exists or can be approximated)
    __device__ __forceinline__ half htanh(const half &x) const {
        return __float2half(tanhf(__half2float(x)));
    }

    // Helper function for bfloat16 precision tanh (assuming it exists or can be approximated)
    __device__ __forceinline__ cuda_bfloat16 htanh(const cuda_bfloat16 &x) const {
        return __float2bfloat16(tanhf(__bfloat162float(x)));
    }
} TanhOp;
} // namespace op::tanh::cuda

#endif // TANH_CUDA_H