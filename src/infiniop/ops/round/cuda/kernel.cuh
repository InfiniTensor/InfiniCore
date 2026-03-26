#ifndef __ROUND_CUDA_KERNEL_H__
#define __ROUND_CUDA_KERNEL_H__

namespace op::round::cuda {

typedef struct RoundOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &x, int decimals) const {
        if constexpr (std::is_same_v<T, half>) {
            if (decimals == 0) {
                return __float2half(nearbyintf(__half2float(x)));
            }
            float scale = powf(10.0f, static_cast<float>(decimals));
            // Multiply in fp16 to match PyTorch behavior
            half scaled = __hmul(x, __float2half(scale));
            float rounded = nearbyintf(__half2float(scaled));
            return __float2half(rounded / scale);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            if (decimals == 0) {
                return __float2bfloat16(nearbyintf(__bfloat162float(x)));
            }
            float scale = powf(10.0f, static_cast<float>(decimals));
            // Multiply in bf16 to match PyTorch behavior
            cuda_bfloat16 scaled = __hmul(x, __float2bfloat16(scale));
            float rounded = nearbyintf(__bfloat162float(scaled));
            return __float2bfloat16(rounded / scale);
        } else if constexpr (std::is_same_v<T, float>) {
            if (decimals == 0) {
                return nearbyintf(x);
            }
            float scale = powf(10.0f, static_cast<float>(decimals));
            return nearbyintf(x * scale) / scale;
        } else {
            // double
            if (decimals == 0) {
                return ::nearbyint(x);
            }
            double scale = pow(10.0, static_cast<double>(decimals));
            return ::nearbyint(x * scale) / scale;
        }
    }
} RoundOp;

} // namespace op::round::cuda

#endif // __ROUND_CUDA_KERNEL_H__