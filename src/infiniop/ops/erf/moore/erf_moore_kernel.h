#ifndef __ERF_MOORE_KERNEL_H__
#define __ERF_MOORE_KERNEL_H__

#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>

namespace op::erf::moore {

typedef struct ErfOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half2>) {
            float x0 = __low2float(x);
            float x1 = __high2float(x);
            return __floats2half2_rn(erff(x0), erff(x1));
        } else if constexpr (std::is_same_v<T, half>) {
            float xf = __half2float(x);
            return __float2half(erff(xf));
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            float xf = __bfloat162float(x);
            return __float2bfloat16_rn(erff(xf));
        } else if constexpr (std::is_same_v<T, float>) {
            return erff(x);
        } else { // double
            return erf(x);
        }
    }
} ErfOp;

} // namespace op::erf::moore

#endif // __ERF_MOORE_KERNEL_H__
