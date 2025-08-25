#ifndef __HARDSWISH_CUDA_H__
#define __HARDSWISH_CUDA_H__

#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace op::hardswish::cuda {

typedef struct HardswishOp {
    static constexpr size_t num_inputs = 1;

    // Hardswish: f(x) = x * clamp(x + 3, 0, 6) / 6
    __device__ __forceinline__ float hswish_f32(float x) const {
        float y = x + 3.0f;
        y = y < 0.0f ? 0.0f : (y > 6.0f ? 6.0f : y);
        return x * (y * (1.0f / 6.0f));
    }

    template <typename T>
    __device__ __forceinline__ T operator()(const T &input) const {
        if constexpr (std::is_same_v<T, half2>) {
            float2 vf = __half22float2(input);
            float2 vr = make_float2(
                hswish_f32(vf.x),
                hswish_f32(vf.y));
            return __float22half2_rn(vr);
        } else if constexpr (std::is_same_v<T, half>) {
            float xf = __half2float(input);
            float yf = hswish_f32(xf);
            return __float2half_rn(yf);
        } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
            float f0 = __bfloat162float(__low2bfloat16(input));
            float f1 = __bfloat162float(__high2bfloat16(input));
            return __floats2bfloat162_rn(hswish_f32(f0), hswish_f32(f1));
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            float xf = __bfloat162float(input);
            return __float2bfloat16_rz(hswish_f32(xf));
        } else if constexpr (std::is_same_v<T, float>) {
            return hswish_f32(input);
        } else if constexpr (std::is_same_v<T, double>) {
            double xd = static_cast<double>(input);
            double yd = xd * (std::fmin(std::fmax(xd + 3.0, 0.0), 6.0) / 6.0);
            return static_cast<T>(yd);
        } else {
            double xd = static_cast<double>(input);
            double yd = xd * (std::fmin(std::fmax(xd + 3.0, 0.0), 6.0) / 6.0);
            return static_cast<T>(yd);
        }
    }
} HardswishOp;

} // namespace op::hardswish::cuda

#endif // __HARDSWISH_CUDA_H__
