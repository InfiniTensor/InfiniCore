#ifndef __COSH_CUDA_KERNEL_H__
#define __COSH_CUDA_KERNEL_H__

namespace op::cosh::cuda {

typedef struct CoshOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half>) {
            float x_f = __half2float(x);
            return __float2half(coshf(x_f));
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            float x_f = __bfloat162float(x);
            return __float2bfloat16(coshf(x_f));
        } else if constexpr (std::is_same_v<T, float>) {
            return coshf(x);
        } else {
            return ::cosh(x);
        }
    }
} CoshOp;

} // namespace op::cosh::cuda

#endif // __COSH_CUDA_KERNEL_H__