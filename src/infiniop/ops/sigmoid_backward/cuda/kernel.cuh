#ifndef __SIGMOID_BACKWARD_CUDA_H__
#define __SIGMOID_BACKWARD_CUDA_H__

#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>

namespace op::sigmoid_backward::cuda {
typedef struct SigmoidBackwardOp {
public:
    static constexpr size_t num_inputs = 2;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &x, const T &grad_out) const {
        if constexpr (std::is_same_v<T, half2>) {
            float2 xf = __half22float2(x);
            float2 gf = __half22float2(grad_out);
            float2 sf;
            sf.x = 1.0f / (1.0f + __expf(-xf.x));
            sf.y = 1.0f / (1.0f + __expf(-xf.y));
            float2 gr;
            gr.x = gf.x * sf.x * (1.0f - sf.x);
            gr.y = gf.y * sf.y * (1.0f - sf.y);
            return __float22half2_rn(gr);
        } else if constexpr (std::is_same_v<T, half>) {
            float xf = __half2float(x);
            float gf = __half2float(grad_out);
            float s = 1.0f / (1.0f + __expf(-xf));
            float gr = gf * s * (1.0f - s);
            return __float2half_rn(gr);
        } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
            float f0 = __bfloat162float(__low2bfloat16(x));
            float f1 = __bfloat162float(__high2bfloat16(x));
            float g0 = __bfloat162float(__low2bfloat16(grad_out));
            float g1 = __bfloat162float(__high2bfloat16(grad_out));
            float s0 = 1.0f / (1.0f + __expf(-f0));
            float s1 = 1.0f / (1.0f + __expf(-f1));
            float r0 = g0 * s0 * (1.0f - s0);
            float r1 = g1 * s1 * (1.0f - s1);
            return __floats2bfloat162_rn(r0, r1);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            float xf = __bfloat162float(x);
            float gf = __bfloat162float(grad_out);
            float s = 1.0f / (1.0f + __expf(-xf));
            float gr = gf * s * (1.0f - s);
            return __float2bfloat16_rn(gr);
        } else if constexpr (std::is_same_v<T, float>) {
            float s = 1.0f / (1.0f + __expf(-x));
            return grad_out * s * (1.0f - s);
        } else if constexpr (std::is_same_v<T, double>) {
            double s = 1.0 / (1.0 + std::exp(-x));
            return grad_out * s * (1.0 - s);
        } else {
            auto s = static_cast<float>(1) / (static_cast<float>(1) + std::exp(-static_cast<float>(x)));
            return static_cast<T>(static_cast<float>(grad_out) * s * (1.0f - s));
        }
    }
} SigmoidBackwardOp;
} // namespace op::sigmoid_backward::cuda

#endif // __SIGMOID_BACKWARD_CUDA_H__
