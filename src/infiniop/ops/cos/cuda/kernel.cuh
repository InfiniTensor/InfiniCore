#ifndef __COS_CUDA_H__
#define __COS_CUDA_H__

#include "../../../elementwise/nvidia/elementwise_nvidia.cuh"
#include <cuda_fp16.h>

namespace op::cos::cuda {
typedef struct CosOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half2>) {
            return h2cos(x);
        } else if constexpr (std::is_same_v<T, half>) {
            return hcos(x);
        } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
            float x0 = __bfloat162float(__low2bfloat16(x));
            float x1 = __bfloat162float(__high2bfloat16(x));
            return __floats2bfloat162_rn(cosf(x0), cosf(x1));
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            return __float2bfloat16_rn(cosf(__bfloat162float(x)));
        } else if constexpr (std::is_same_v<T, float>) {
            return __cosf(x);
        } else {
            return std::cos(x);
        }
    }
} CosOp;
} // namespace op::cos::cuda

#endif // __COS_CUDA_H__
