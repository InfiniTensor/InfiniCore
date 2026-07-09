#ifndef __QUICKGELU_CUDA_H__
#define __QUICKGELU_CUDA_H__

#include <cmath>

namespace op::quickgelu::cuda {

typedef struct QuickGeluOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        // quickgelu(x) = x * sigmoid(1.702 * x) = x / (1 + exp(-1.702 x))
        constexpr float alpha = 1.702f;
        if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            float x_f = __bfloat162float(x);
            float result = x_f / (1.0f + expf(-alpha * x_f));
            return __float2bfloat16(result);
        } else if constexpr (std::is_same_v<T, half>) {
            float x_f = __half2float(x);
            float result = x_f / (1.0f + expf(-alpha * x_f));
            return __float2half(result);
        } else if constexpr (std::is_same_v<T, float>) {
            return x / (1.0f + expf(-alpha * x));
        } else {
            return x / (1.0 + exp(-static_cast<double>(alpha) * x));
        }
    }
} QuickGeluOp;

} // namespace op::quickgelu::cuda

#endif // __QUICKGELU_CUDA_H__
