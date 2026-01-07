#ifndef __LOG_CUDA_H__
#define __LOG_CUDA_H__

#include "../../../elementwise/nvidia/elementwise_nvidia.cuh"
#include <cuda_fp16.h>

namespace op::log::cuda {
typedef struct LogOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half2>) {
            return h2log(x);
        } else if constexpr (std::is_same_v<T, half>) {
            return __float2half(__logf(__half2float(x)));
        } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
            float x0 = __bfloat162float(__low2bfloat16(x));
            float x1 = __bfloat162float(__high2bfloat16(x));
            return __floats2bfloat162_rn(logf(x0), logf(x1));
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            return __float2bfloat16_rn(logf(__bfloat162float(x)));
        } else if constexpr (std::is_same_v<T, float>) {
            return __logf(x);
        } else {
            return std::log(x);
        }
    }
} LogOp;
} // namespace op::log::cuda

#endif // __LOG_CUDA_H__
