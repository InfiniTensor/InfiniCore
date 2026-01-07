#ifndef __SIGN_CUDA_H__
#define __SIGN_CUDA_H__

#include "../../../elementwise/nvidia/elementwise_nvidia.cuh"
#include <cuda_fp16.h>

namespace op::sign::cuda {
typedef struct SignOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half2>) {
            const auto lt_mask = __hlt2(x, __floats2half2_rn(0.0f, 0.0f));
            return __hadd2(__hneg2(lt_mask), __hsub2(__floats2half2_rn(1.0f, 1.0f), lt_mask));
        } else if constexpr (std::is_same_v<T, half>) {
            return x > half(0) ? half(1) : (x == half(0) ? half(0) : half(-1));
        } else {
            return x > T(0) ? T(1) : (x == T(0) ? T(0) : T(-1));
        }
    }
} SignOp;
} // namespace op::sign::cuda

#endif // __SIGN_CUDA_H__
