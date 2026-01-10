#ifndef __EQUAL_MOORE_KERNEL_H__
#define __EQUAL_MOORE_KERNEL_H__

#include <type_traits>

namespace op::equal::moore {

typedef struct EqualOp {
public:
    static constexpr size_t num_inputs = 2;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        if constexpr (std::is_same_v<T, half>) {
            float af = __half2float(a);
            float bf = __half2float(b);
            return __float2half(af == bf ? 1.0f : 0.0f);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            float af = __bfloat162float(a);
            float bf = __bfloat162float(b);
            return __float2bfloat16_rn(af == bf ? 1.0f : 0.0f);
        } else {
            return static_cast<T>(a == b);
        }
    }

    template <typename Tout, typename Tin0, typename Tin1>
    __device__ __forceinline__ Tout operator()(const Tin0 &a, const Tin1 &b) const {
        static_assert(std::is_same_v<Tin0, Tin1>, "EqualOp expects identical input dtypes");
        bool eq = false;
        if constexpr (std::is_same_v<Tin0, half>) {
            eq = __half2float(a) == __half2float(b);
        } else if constexpr (std::is_same_v<Tin0, cuda_bfloat16>) {
            eq = __bfloat162float(a) == __bfloat162float(b);
        } else {
            eq = a == b;
        }
        return static_cast<Tout>(eq);
    }
} EqualOp;

} // namespace op::equal::moore

#endif // __EQUAL_MOORE_KERNEL_H__
