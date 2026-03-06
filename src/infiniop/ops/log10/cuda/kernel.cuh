#pragma once
#include <cmath> // 包含 log10f, log10, log, logf 等
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <type_traits>

namespace op::cuda {

// 移除 high_precision_log10f 避免混淆，让 Log10Op 直接实现逻辑。

struct Log10Op {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, float>) {
            // For F32: compute via F64 for improved accuracy.
            return (float)log10((double)x);
        } else if constexpr (std::is_same_v<T, double>) {
            return log10(x);
        } else {
            // For F16/BF16: promote to float, compute, then cast back.
            return (T)(float)log10((double)(float)x);
        }
    }
};

} // namespace op::cuda
