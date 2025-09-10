#ifndef __SOFTPLUS_CUDA_H__
#define __SOFTPLUS_CUDA_H__

namespace op::softplus::cuda {
typedef struct SoftplusOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half>) {
            // promote to float for stability, then cast back
            float xf = __half2float(x);
            float out = (xf > 20.0f) ? xf : log1pf(expf(xf));
            return __float2half(out);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            float xf = __bfloat162float(x);
            float out = (xf > 20.0f) ? xf : log1pf(expf(xf));
            return __float2bfloat16(out);
        } else if constexpr (std::is_same_v<T, half2>) {
            // process as two lanes
            float2 xf = __half22float2(x);
            xf.x = (xf.x > 20.0f) ? xf.x : log1pf(expf(xf.x));
            xf.y = (xf.y > 20.0f) ? xf.y : log1pf(expf(xf.y));
            return __floats2half2_rn(xf.x, xf.y);
        } else {
            // default: float, double, etc.
            return (x > T(20)) ? x : log1p(exp(x));
        }
    }
} SoftplusOp;
} // namespace op::softplus::cuda

#endif // __SOFTPLUS_CUDA_H__
