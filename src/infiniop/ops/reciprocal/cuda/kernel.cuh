#ifndef __RECIPROCAL_CUDA_H__
#define __RECIPROCAL_CUDA_H__

#include <type_traits>

namespace op::reciprocal::cuda {
typedef struct ReciprocalOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half2>) {
            float2 vf = __half22float2(x);
            vf.x = 1.0f / vf.x;
            vf.y = 1.0f / vf.y;
            return __float22half2_rn(vf);
        } else if constexpr (std::is_same_v<T, half>) {
            return __float2half(1.0f / __half2float(x));
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            return __float2bfloat16_rn(1.0f / __bfloat162float(x));
        } else if constexpr (std::is_same_v<T, float>) {
            return 1.0f / x;
        } else {
            return static_cast<T>(1) / x;
        }
    }
} ReciprocalOp;
} // namespace op::reciprocal::cuda

#endif // __RECIPROCAL_CUDA_H__
