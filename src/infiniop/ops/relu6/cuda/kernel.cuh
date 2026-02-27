#pragma once
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <type_traits>
#include <algorithm>

namespace op::cuda {

template <typename T>
struct Relu6Op {
    __device__ __forceinline__ T operator()(T x) const {
        if constexpr (std::is_same_v<T, float>) {
            return fminf(fmaxf(x, 0.0f), 6.0f);
        } else if constexpr (std::is_same_v<T, double>) {
            return std::min(std::max(x, 0.0), 6.0);
        } else {
            // For F16/BF16: promote to float, compute, then cast back
            float xf = static_cast<float>(x);
            return static_cast<T>(fminf(fmaxf(xf, 0.0f), 6.0f));
        }
    }
};

} // namespace op::cuda
