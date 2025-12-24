#pragma once
#include <cuda_runtime.h>
#include <type_traits>

namespace op::cuda {

template <typename T>
struct BitwiseRightShiftOp {
    __device__ __forceinline__ T operator()(T x, T shift) const {
        return x >> shift;
    }
};

} // namespace op::cuda
