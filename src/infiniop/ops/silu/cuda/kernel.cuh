#ifndef __SILU_CUDA_H__
#define __SILU_CUDA_H__

namespace op::silu::cuda {
typedef struct SiluOp {
private:
    template <typename T>
    __device__ __forceinline__ T sigmoid(const T &x) const {
        if constexpr (std::is_same_v<T, half2>) {
            return h2rcp(__hadd2(make_half2(1, 1), h2exp(__hneg2(x))));
        } else if constexpr (std::is_same_v<T, half>) {
            return hrcp(__hadd(half(1.f), __float2half(__expf(__half2float(__hneg(x))))));
        } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
            float x0 = __bfloat162float(__low2bfloat16(x));
            float x1 = __bfloat162float(__high2bfloat16(x));
            float sig0 = __frcp_rn(__fadd_rn(1.0f, __expf(-x0)));
            float sig1 = __frcp_rn(__fadd_rn(1.0f, __expf(-x1)));
            return __floats2bfloat162_rn(sig0, sig1);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            float xf = __bfloat162float(x);
            return __float2bfloat16_rn(__frcp_rn(__fadd_rn(1.0f, __expf(-xf))));
        } else if constexpr (std::is_same_v<T, float>) {
            return __frcp_rn(__fadd_rn(1, __expf(-x)));
        } else {
            return 1 / (1 + std::exp(x));
        }
    }

public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half2>) {
            return __hmul2(x, sigmoid(x));
        } else if constexpr (std::is_same_v<T, half>) {
            return __hmul(x, sigmoid(x));
        } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
            cuda_bfloat162 sig = sigmoid(x);

            float x0 = __bfloat162float(__low2bfloat16(x));
            float x1 = __bfloat162float(__high2bfloat16(x));
            float sig0 = __bfloat162float(__low2bfloat16(sig));
            float sig1 = __bfloat162float(__high2bfloat16(sig));

            float res0 = __fmul_rn(x0, sig0);
            float res1 = __fmul_rn(x1, sig1);
            return __floats2bfloat162_rn(res0, res1);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            cuda_bfloat16 sig = sigmoid(x);

            float xf = __bfloat162float(x);
            float sigf = __bfloat162float(sig);

            float res = __fmul_rn(xf, sigf);
            return __float2bfloat16_rn(res);
        } else if constexpr (std::is_same_v<T, float>) {
            return __fmul_rn(x, sigmoid(x));
        } else {
            return x * sigmoid(x);
        }
    };
} SiluOp;
} // namespace op::silu::cuda

#endif // __silu_CUDA_H__
