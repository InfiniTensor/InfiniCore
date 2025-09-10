#ifndef __OR_CUDA_H__
#define __OR_CUDA_H__

namespace op::or_op::cuda {
typedef struct OrOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        // Logical OR operation for CUDA
        if constexpr (std::is_same_v<T, bool>) {
            return a || b;
        } else if constexpr (std::is_same_v<T, half2>) {
            // For half2, we need to handle each component
            half2 zero = __float2half2_rn(0.0f);
            half2 one = __float2half2_rn(1.0f);
            half2 a_bool = __hne2(a, zero) ? one : zero;
            half2 b_bool = __hne2(b, zero) ? one : zero;
            // OR operation: result is 1 if either a_bool or b_bool is non-zero
            half2 result;
            result.x = (__hne(a_bool.x, __float2half(0.0f)) || __hne(b_bool.x, __float2half(0.0f))) ? __float2half(1.0f) : __float2half(0.0f);
            result.y = (__hne(a_bool.y, __float2half(0.0f)) || __hne(b_bool.y, __float2half(0.0f))) ? __float2half(1.0f) : __float2half(0.0f);
            return result;
        } else if constexpr (std::is_same_v<T, half>) {
            half zero = __float2half(0.0f);
            half one = __float2half(1.0f);
            return (__hne(a, zero) || __hne(b, zero)) ? one : zero;
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            cuda_bfloat16 zero = __float2bfloat16(0.0f);
            cuda_bfloat16 one = __float2bfloat16(1.0f);
            return (__hne(a, zero) || __hne(b, zero)) ? one : zero;
        } else if constexpr (std::is_floating_point_v<T>) {
            return static_cast<T>((a != static_cast<T>(0)) || (b != static_cast<T>(0)));
        } else {
            // Integer types
            return static_cast<T>((a != static_cast<T>(0)) || (b != static_cast<T>(0)));
        }
    }
} OrOp;
} // namespace op::or_op::cuda

#endif // __OR_CUDA_H__