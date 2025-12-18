#ifndef __GELU_MOORE_KERNEL_H__
#define __GELU_MOORE_KERNEL_H__

/*
 * This file contains the GELU operation implementation for the MUSA backend.
 *
 * It uses the 'op::gelu::cuda' namespace to maintain a consistent code structure
 * and interface with the CUDA implementation, ensuring code alignment across different
 * hardware platforms.
 */

#include <cmath>

namespace op::gelu::moore {

typedef struct GeluOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {

        if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            float x_f = __bfloat162float(x);
            float result = 0.5f * x_f * (1.0f + erff(x_f / sqrtf(2.0f)));

            return __float2bfloat16(result);
        } else if constexpr (std::is_same_v<T, half>) {
            float x_f = __half2float(x);
            float result = 0.5f * x_f * (1.0f + erff(x_f / sqrtf(2.0f)));

            return __float2half(result);
        } else if constexpr (std::is_same_v<T, float>) {

            return 0.5f * x * (1.0f + erff(x / sqrtf(2.0f)));
        } else {
            return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
        }
    }
} GeluOp;

} // namespace op::gelu::moore

#endif // __GELU_MOORE_KERNEL_H__