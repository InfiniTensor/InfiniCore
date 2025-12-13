#ifndef __ERFC_MOORE_KERNEL_H__
#define __ERFC_MOORE_KERNEL_H__

#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>

namespace op::erfc::moore {

typedef struct ErfcOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half2>) {
            float x0 = __low2float(x);
            float x1 = __high2float(x);
            return __floats2half2_rn(erfcf(x0), erfcf(x1));
        } else if constexpr (std::is_same_v<T, half>) {
            float xf = __half2float(x);
            return __float2half(erfcf(xf));
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            float xf = __bfloat162float(x);
            return __float2bfloat16_rn(erfcf(xf));
        } else if constexpr (std::is_same_v<T, float>) {
            return erfcf(x);
        } else { // double
            return erfc(x);
        }
    }
} ErfcOp;

} // namespace op::erfc::moore

#endif // __ERFC_MOORE_KERNEL_H__
