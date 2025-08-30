#ifndef __SILU_CUDA_H__
#define __SILU_CUDA_H__

#include <cmath>

namespace op::silu::cuda {
typedef struct SiluOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half>) {
            float fx = __half2float(x);
            float sigmoid_x = 1.0f / (1.0f + expf(-fx));
            return __float2half(fx * sigmoid_x);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            float fx = __bfloat162float(x);
            float sigmoid_x = 1.0f / (1.0f + expf(-fx));
            return __float2bfloat16(fx * sigmoid_x);
        } else if constexpr (std::is_same_v<T, float>) {
            float sigmoid_x = 1.0f / (1.0f + expf(-x));
            return x * sigmoid_x;
        } else if constexpr (std::is_same_v<T, double>) {
            double sigmoid_x = 1.0 / (1.0 + exp(-x));
            return x * sigmoid_x;
        } else {
            // Fallback for other types
            T sigmoid_x = T(1) / (T(1) + exp(-x));
            return x * sigmoid_x;
        }
    }
} SiluOp;
} // namespace op::silu::cuda

#endif // __SILU_CUDA_H__