#ifndef __AND_CUDA_H__
#define __AND_CUDA_H__

namespace op::and_op::cuda {
typedef struct AndOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        // Logical AND operation for CUDA
        if constexpr (std::is_same_v<T, bool>) {
            return a && b;
        } else if constexpr (std::is_same_v<T, half2>) {
            // For half2, we need to handle each component
            half2 zero = __float2half2_rn(0.0f);
            half2 one = __float2half2_rn(1.0f);
            half2 a_bool = __hne2(a, zero) ? one : zero;
            half2 b_bool = __hne2(b, zero) ? one : zero;
            return __hmul2(a_bool, b_bool);
        } else if constexpr (std::is_same_v<T, half>) {
            half zero = __float2half(0.0f);
            half one = __float2half(1.0f);
            return (__hne(a, zero) && __hne(b, zero)) ? one : zero;
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            cuda_bfloat16 zero = __float2bfloat16(0.0f);
            cuda_bfloat16 one = __float2bfloat16(1.0f);
            return (__hne(a, zero) && __hne(b, zero)) ? one : zero;
        } else if constexpr (std::is_floating_point_v<T>) {
            return static_cast<T>((a != static_cast<T>(0)) && (b != static_cast<T>(0)));
        } else {
            // Integer types
            return static_cast<T>((a != static_cast<T>(0)) && (b != static_cast<T>(0)));
        }
    }
} AndOp;
} // namespace op::and_op::cuda

#endif // __AND_CUDA_H__