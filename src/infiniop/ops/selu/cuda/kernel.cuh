#pragma once
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace op::cuda {

// SELU constants
constexpr float SELU_ALPHA = 1.6732632423543772848170429916717f;
constexpr float SELU_SCALE = 1.0507009873554804934193349852946f;

template <typename T>
struct SeluOp {
    __device__ __forceinline__ T operator()(T x) const {
        if constexpr (std::is_same_v<T, float>) {
            return x > 0.0f ? SELU_SCALE * x : SELU_SCALE * SELU_ALPHA * (expf(x) - 1.0f);
        } else if constexpr (std::is_same_v<T, double>) {
            return x > 0.0 ? static_cast<double>(SELU_SCALE) * x : static_cast<double>(SELU_SCALE) * static_cast<double>(SELU_ALPHA) * (exp(x) - 1.0);
        } else {
            // For F16/BF16: promote to float, compute, then cast back
            float xf = static_cast<float>(x);
            float result = xf > 0.0f ? SELU_SCALE * xf : SELU_SCALE * SELU_ALPHA * (expf(xf) - 1.0f);
            return static_cast<T>(result);
        }
    }
};

} // namespace op::cuda
