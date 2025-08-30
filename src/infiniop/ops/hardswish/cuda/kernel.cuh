#ifndef HARDSWISH_CUDA_H
#define HARDSWISH_CUDA_H

namespace op::hardswish::cuda {

typedef struct HardSwishOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half>) {
            float xf = __half2float(x);
            float t = fminf(6.0f, fmaxf(0.0f, xf + 3.0f)) * (1.0f / 6.0f);
            return __float2half(xf * t);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            float xf = __bfloat162float(x);
            float t = fminf(6.0f, fmaxf(0.0f, xf + 3.0f)) * (1.0f / 6.0f);
            return __float2bfloat16(xf * t);
        } else if constexpr (std::is_same_v<T, float>) {
            float t = fminf(6.0f, fmaxf(0.0f, x + 3.0f)) * (1.0f / 6.0f);
            return x * t;
        } else if constexpr (std::is_same_v<T, double>) {
            double t = fmin(6.0, fmax(0.0, x + 3.0)) * (1.0 / 6.0);
            return x * t;
        } else {
            float xf = static_cast<float>(x);
            float t = fminf(6.0f, fmaxf(0.0f, xf + 3.0f)) * (1.0f / 6.0f);
            return static_cast<T>(xf * t);
        }
    }
} HardSwishOp;

} // namespace op::hardswish::cuda

#endif // HARDSWISH_CUDA_H
