#pragma once
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <type_traits>

namespace op::cuda {

template <typename T>
struct Log1pOp {
    __device__ __forceinline__ T operator()(T x) const {
        if constexpr (std::is_same_v<T, float>) {
            // Use double precision for better accuracy
            return (float)log1p((double)x);
        } else if constexpr (std::is_same_v<T, double>) {
            return log1p(x);
        } else {
            // For F16/BF16: promote to float, compute, then cast back
            return (T)(float)log1p((double)(float)x);
        }
    }
};

} // namespace op::cuda
