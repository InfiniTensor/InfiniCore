#ifndef __COS_CUDA_H__
#define __COS_CUDA_H__

#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace op::cos::cuda {
typedef struct CosOp {
    static constexpr size_t num_inputs = 1;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &input) const {
        auto cos_f32 = [] __device__(float x) {
            double xd = static_cast<double>(x);
            double yd = std::cos(xd);
            return static_cast<float>(yd);
        };

        if constexpr (std::is_same_v<T, half2>) {
            float2 vf = __half22float2(input);
            float2 vr = make_float2(
                cos_f32(vf.x),
                cos_f32(vf.y));
            return __float22half2_rn(vr);
        } else if constexpr (std::is_same_v<T, half>) {
            float xf = __half2float(input);
            float yf = cos_f32(xf);
            return __float2half_rn(yf);
        } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
            float f0 = __bfloat162float(__low2bfloat16(input));
            float f1 = __bfloat162float(__high2bfloat16(input));
            return __floats2bfloat162_rz(cos_f32(f0), cos_f32(f1));
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            float xf = __bfloat162float(input);
            return __float2bfloat16_rz(cos_f32(xf));
        } else if constexpr (std::is_same_v<T, float>) {
            return cos_f32(input);
        } else if constexpr (std::is_same_v<T, double>) {
            return std::cos(input);
        } else {
            return std::cos(input);
        }
    }
} CosOp;
} // namespace op::cos::cuda

#endif // __COS_CUDA_H__
