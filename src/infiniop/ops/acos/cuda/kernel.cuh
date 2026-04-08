#ifndef __ACOS_CUDA_H__
#define __ACOS_CUDA_H__

#include <cmath>
#include <math.h>
#include <type_traits>

namespace op::acos::cuda {

// ----------------------
// float kernel (F32)
// ----------------------
template <typename T>
__device__ __forceinline__ T acos_impl(T val);

template <>
__device__ __forceinline__ float acos_impl<float>(float val) {
    return ::acosf(val);
}

// ----------------------
// half kernel (F16)
// ----------------------
template <>
__device__ __forceinline__ half acos_impl<half>(half val) {
    float f = __half2float(val);
    return __float2half(::acosf(f));
}

// ----------------------
// half2 kernel (F16x2 vectorized)
// ----------------------
template <>
__device__ __forceinline__ half2 acos_impl<half2>(half2 val) {
    float2 f = __half22float2(val);
    f.x = ::acosf(f.x);
    f.y = ::acosf(f.y);
    return __float22half2_rn(f);
}

// ----------------------
// bfloat16 kernel (BF16)
// ----------------------
template <>
__device__ __forceinline__ cuda_bfloat16 acos_impl<cuda_bfloat16>(cuda_bfloat16 val) {
    float f = __bfloat162float(val);
    return __float2bfloat16(::acosf(f));
}

template <>
__device__ __forceinline__ double acos_impl<double>(double val) {
    return ::acos(val);
}

// ----------------------
// Fallback kernel
// ----------------------
template <typename T>
__device__ __forceinline__ T acos_impl(T val) {
    return static_cast<T>(::acos(static_cast<double>(val)));
}

// ----------------------
// AcosOp struct
// ----------------------
struct AcosOp {
    static constexpr size_t num_inputs = 1;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &a) const {
        return acos_impl(a);
    }
};

} // namespace op::acos::cuda

#endif // __ACOS_CUDA_H__
