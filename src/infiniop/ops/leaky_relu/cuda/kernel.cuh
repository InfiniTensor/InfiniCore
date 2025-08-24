#ifndef __LEAKY_RELU_CUDA_H__
#define __LEAKY_RELU_CUDA_H__


namespace op::leaky_relu::cuda {
typedef struct LeakyReluOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x, float negative_slope) const {
        if constexpr (std::is_same_v<T, half2>) {
            const half2 zero = __float2half2_rn(0.0f);
            const half2 slope = __float2half2_rn(negative_slope);
            return __hge2(x, zero) ? x : __hmul2(slope, x);
        } else if constexpr (std::is_same_v<T, half>) {
            // Resolution 1
            return __hge(x, __float2half(0.0f)) ? x : __hmul(__float2half(negative_slope), x);
            // Resolution 2
            // const half zero = __float2half_rn(0.0f);
            // const half slope = __float2half_rn(negative_slope);
            // return __hge(x, zero) ? x : __hmul(slope, x);
            // Resolution 3
            // float xf = __half2float(x);
            // float res = xf >= 0.0f ? xf : negative_slope * xf;
            // return __float2half(res);
        }  else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            return __bfloat162float(x) >= 0.0f ? x : __hmul(__float2bfloat16(negative_slope), x);
        } else if constexpr (std::is_same_v<T, float>) {
            return x>=0.0f ? x : __fmul_rn(negative_slope, x);
        } else {
            return x>=0 ? x : negative_slope * x;
        }
    }
} LeakyReluOp;
} // namespace op::leaky_relu::cuda

#endif // __LEAKY_RELU_CUDA_H__
