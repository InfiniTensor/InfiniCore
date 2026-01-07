#ifndef __RECIPROCAL_CUDA_H__
#define __RECIPROCAL_CUDA_H__

#include "../../../elementwise/nvidia/elementwise_nvidia.cuh"
#include <cuda_fp16.h>

namespace op::reciprocal::cuda {
typedef struct ReciprocalOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half2>) {
            return h2rcp(x);
        } else if constexpr (std::is_same_v<T, half>) {
            return hrcp(x);
        } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
            float x0 = __bfloat162float(__low2bfloat16(x));
            float x1 = __bfloat162float(__high2bfloat16(x));
            return __floats2bfloat162_rn(__frcp_rn(x0), __frcp_rn(x1));
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            return __float2bfloat16_rn(__frcp_rn(__bfloat162float(x)));
        } else if constexpr (std::is_same_v<T, float>) {
            return __frcp_rn(x);
        } else {
            return T(1) / x;
        }
    }
} ReciprocalOp;
} // namespace op::reciprocal::cuda

#endif // __RECIPROCAL_CUDA_H__
