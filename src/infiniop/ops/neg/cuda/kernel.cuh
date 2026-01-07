#ifndef __NEG_CUDA_H__
#define __NEG_CUDA_H__

#include <cuda_fp16.h>

namespace op::neg::cuda {
typedef struct NegOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half2>) {
            return __hneg2(x);
        } else if constexpr (std::is_same_v<T, half>) {
            return __hneg(x);
        } else {
            return -x;
        }
    }
} NegOp;
} // namespace op::neg::cuda

#endif // __NEG_CUDA_H__
