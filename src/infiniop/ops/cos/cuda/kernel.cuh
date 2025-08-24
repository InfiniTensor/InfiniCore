#ifndef __COS_CUDA_H__
#define __COS_CUDA_H__

#include <cmath>

namespace op::cos::cuda {
typedef struct CosOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a) const {
        if constexpr (std::is_same_v<T, half2>) {
            return hsin2(cosf(a));
        } else if constexpr (std::is_same_v<T, cuda_bfloat16> || std::is_same_v<T, half>) {
            return hcos(a);
        } else {
            // fallback for other types
            return ::cos(a);
        }
    }
} CosOp;
} // namespace op::cos::cuda

#endif // __COS_CUDA_H__