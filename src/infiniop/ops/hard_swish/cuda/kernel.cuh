#ifndef __HARD_SWISH_CUDA_H__
#define __HARD_SWISH_CUDA_H__


namespace op::hard_swish::cuda {
typedef struct HardSwishOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a) const {
        if constexpr (std::is_same_v<T, half>) {
            half three = __float2half(3.0f);
            half six = __float2half(6.0f);
            half zero = __float2half(0.0f);
            half tmp = __hadd(a, three);                   
            half clipped = __hmin(__hmax(tmp, zero), six);  
            return __hmul(a, __hdiv(clipped, six));         
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            float af = __bfloat162float(a);
            float relu6 = fminf(fmaxf(af + 3.0f, 0.0f), 6.0f);
            float result = af * relu6 / 6.0f;
            return __float2bfloat16(result);
        } else if constexpr (std::is_same_v<T, float>) {
            float relu6 = fminf(fmaxf(a + 3.0f, 0.0f), 6.0f);
            return a * relu6 / 6.0f;
        } else if constexpr (std::is_same_v<T, double>) {
            double relu6 = fmin(fmax(a + 3.0, 0.0), 6.0);
            return a * relu6 / 6.0;
        } else {
            auto relu6 = std::min(std::max(a + T(3), T(0)), T(6));
            return a * relu6 / T(6);
        }
    }
} HardSwishOp;
} // namespace op::hard_swish::cuda

#endif // __HARD_SWISH_CUDA_H__
