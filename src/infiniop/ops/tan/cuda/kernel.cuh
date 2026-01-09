#ifndef __TAN_CUDA_H__
#define __TAN_CUDA_H__

#include "../../../elementwise/nvidia/elementwise_nvidia.cuh"
#include <cmath>
#include <cuda_fp16.h>

#define TAN_THRESHOLD 15000

namespace op::tan::cuda {
typedef struct TanOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half2>) {
            return h2sin(x) / h2cos(x);
        } else if constexpr (std::is_same_v<T, half>) {
            float tan_f = __tanf(__half2float(x));
            if (std::fabs(tan_f) > TAN_THRESHOLD) {
                return __float2half(tanf(__half2float(x)));
            }
            return __float2half(tan_f);
        } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
            float x0 = __bfloat162float(__low2bfloat16(x));
            float x1 = __bfloat162float(__high2bfloat16(x));
            float tan_f0 = __tanf(x0);
            float tan_f1 = __tanf(x1);
            if (std::fabs(tan_f0) > TAN_THRESHOLD) {
                tan_f0 = tanf(x0);
            }
            if (std::fabs(tan_f1) > TAN_THRESHOLD) {
                tan_f1 = tanf(x1);
            }
            return __floats2bfloat162_rn(tan_f0, tan_f1);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            float tan_f = __tanf(__bfloat162float(x));
            if (std::fabs(tan_f) > TAN_THRESHOLD) {
                return __float2bfloat16_rn(tanf(__bfloat162float(x)));
            }
            return __float2bfloat16_rn(tan_f);
        } else if constexpr (std::is_same_v<T, float>) {
            float tan_f = __tanf(x);
            if (std::fabs(tan_f) > TAN_THRESHOLD) {
                return tanf(x);
            }
            return tan_f;
        } else {
            return std::tan(x);
        }
    }
} TanOp;
} // namespace op::tan::cuda

#endif // __TAN_CUDA_H__
