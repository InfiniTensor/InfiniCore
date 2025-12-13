#pragma once
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace op::cuda {

template <typename T>
struct PreluOp {
    __device__ __forceinline__ T operator()(T x, T weight) const {
        return x > 0 ? x : weight * x;
    }
};

} // namespace op::cuda
