#ifndef __TANH_CUDA_H__
#define __TANH_CUDA_H__

#include <cmath>
// #include <cuda_bf16.h>
// #include <cuda_fp16.h>
// #include <cuda_runtime.h>

namespace op::tanh::cuda {
typedef struct TanhOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half2>) {
            return htanh2(x);
        } 
        // else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, cuda_bfloat16>) {
        //     return htanh(x);
        // }
        // else if constexpr (std::is_same_v<T, float>) {
        //     return ftanh_rd(x);
        // }
        else {
            return std::tanh(x);
        }
    }
} TanhOp;
} // namespace op::tanh::cuda

#endif // __TANH_CUDA_H__ 