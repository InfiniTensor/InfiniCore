#ifndef __SQRT_CUDA_H__
#define __SQRT_CUDA_H__

// #include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace op::sqrt::cuda {
typedef struct SqrtOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half2>) {
            float2 xf = __half22float2(x);
            return __floats2half2_rn(sqrtf(xf.x), sqrtf(xf.y));
        } else if constexpr (std::is_same_v<T, half>) {
            // FP16, convert to float first
            float xf = __half2float(x);
            return __float2half(sqrtf(xf));
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            // BF16, convert to float first
            float xf = __bfloat162float(x);
            return __float2bfloat16(sqrtf(xf));
        } else if constexpr (std::is_same_v<T, float>) {
            return sqrtf(x);
        } else if constexpr (std::is_same_v<T, double>) {
            return sqrtf(x);
        } else {
            return sqrtf(x);
        }
    }
} SqrtOp;
} // namespace op::sqrt::cuda

#endif // __SQRT_CUDA_H__