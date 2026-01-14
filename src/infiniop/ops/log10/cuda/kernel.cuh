#pragma once
#include <cmath> // 包含 log10f, log10, log, logf 等
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <type_traits>

namespace op::cuda {

// 移除 high_precision_log10f 避免混淆，让 Log10Op 直接实现逻辑。

template <typename T>
struct Log10Op {
    __device__ __forceinline__ T operator()(T x) const {
        if constexpr (std::is_same_v<T, float>) {
            // ----------------------------------------------------
            // 针对 F32：回归到最直接的 F64 log10 转换路径
            // ----------------------------------------------------
            return (float)log10((double)x);

        } else if constexpr (std::is_same_v<T, double>) {
            // F64
            return log10(x);

        } else {
            // 针对 F16/BF16：提升为 F32，然后使用 F64 log10 转换路径
            return (T)(float)log10((double)(float)x);
        }
    }
};

} // namespace op::cuda