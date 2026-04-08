#ifndef __TAN_CUDA_H__
#define __TAN_CUDA_H__

#include <cmath>
#include <type_traits>

namespace op::tan::cuda {

typedef struct TanOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half2>) {
            float2 vf = __half22float2(x);
            vf.x = ::tanf(vf.x);
            vf.y = ::tanf(vf.y);
            return __float22half2_rn(vf);
        } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
            float f0 = __bfloat162float(__low2bfloat16(x));
            float f1 = __bfloat162float(__high2bfloat16(x));
            return __floats2bfloat162_rn(::tanf(f0), ::tanf(f1));
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            // BF16
            const float x_f = __bfloat162float(x);
            return __float2bfloat16_rn(::tanf(x_f));
        } else if constexpr (std::is_same_v<T, half>) {
            // FP16
            const float x_f = __half2float(x);
            return __float2half(::tanf(x_f));
        } else if constexpr (std::is_same_v<T, float>) {
            // FP32
            return ::tanf(x);
        } else if constexpr (std::is_same_v<T, double>) {
            return ::tan(x);
        } else {
            return static_cast<T>(::tan(static_cast<double>(x)));
        }
    }
} TanOp;

} // namespace op::tan::cuda

#endif // __TAN_CUDA_H__
