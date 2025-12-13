#pragma once
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace op::cuda {

template <typename T>
struct ErfcOp {
    __device__ __forceinline__ T operator()(T x) const {
        if constexpr (std::is_same_v<T, float>) {
            return erfcf(x);
        } else if constexpr (std::is_same_v<T, double>) {
            return erfc(x);
        } else {
            // For F16/BF16: promote to float, compute, then cast back
            float xf = static_cast<float>(x);
            return static_cast<T>(erfcf(xf));
        }
    }
};

} // namespace op::cuda
