#ifndef __EQUAL_CUDA_H__
#define __EQUAL_CUDA_H__

namespace op::equal_op::cuda {
typedef struct EqualOp {
public:
    static constexpr size_t num_inputs = 2;

    template <typename T, typename Ta, typename Tb>
    __device__ __forceinline__ T operator()(const Ta &a, const Tb &b) const {
        if constexpr (std::is_same_v<Ta, half2>) {
            // For half2, compare each component separately
            half a_low = __low2half(a);
            half a_high = __high2half(a);
            half b_low = __low2half(b);
            half b_high = __high2half(b);
            return __heq(a_low, b_low) && __heq(a_high, b_high);
        } else if constexpr (std::is_same_v<Ta, half>) {
            return static_cast<bool>(__heq(a, b));
        } else if constexpr (std::is_same_v<Ta, cuda_bfloat16>) {
            return static_cast<bool>(__heq(a, b));
        } else if constexpr (std::is_same_v<Ta, bool>) {
            // Explicit handling for bool type
            return static_cast<bool>(a == b);
        } else if constexpr (std::is_same_v<Ta, int8_t> || std::is_same_v<Ta, uint8_t>) {
            // For 8-bit integers (which might be used for bool storage)
            return static_cast<bool>(a == b);
        } else {
            // For all other types (float, double, int, etc.)
            return static_cast<bool>(a == b);
        }
    }
} EqualOp;
} // namespace op::equal_op::cuda

#endif // __EQUAL_CUDA_H__