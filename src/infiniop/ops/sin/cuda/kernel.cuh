#ifndef __SIN_CUDA_H__
#define __SIN_CUDA_H__

#include <cmath>
// #include <cuda_bf16.h>
// #include <cuda_fp16.h>
// #include <cuda_runtime.h>

namespace op::sin::cuda {
typedef struct SinOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half2>) {
            return hsin2(x);
        } else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, cuda_bfloat16>) {
            return hsin(x);
        } 
        // else if constexpr (std::is_same_v<T, float>) {
        //     return fsin_rd(x);
        // } 
        else {
            return std::sin(x);
        }
        return std::sin(x);
    }
} SinOp;
} // namespace op::sin::cuda

#endif // __SIN_CUDA_H__ 