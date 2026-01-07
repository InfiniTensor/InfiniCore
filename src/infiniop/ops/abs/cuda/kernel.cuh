#ifndef __ABS_CUDA_H__
#define __ABS_CUDA_H__

#include <cmath>
#include <cuda_fp16.h>

namespace op::abs::cuda {
typedef struct AbsOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half2>) {
            return __habs2(x);
        } else if constexpr (std::is_same_v<T, half>) {
            return __habs(x);
        } else if constexpr (std::is_floating_point_v<T>) {
            return std::fabs(x);
        } else {
            return std::abs(x);
        }
    }
} AbsOp;
} // namespace op::abs::cuda

#endif // __ABS_CUDA_H__
