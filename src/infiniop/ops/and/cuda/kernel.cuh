#ifndef __AND_CUDA_H__
#define __AND_CUDA_H__
// #include <cuda_runtime.h>
// #include <math.h>

namespace op::op_and::cuda {
typedef struct AndOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        if constexpr (std::is_same<T, bool>::value)
            return a && b;
        else
            return bool(a) && bool(b);
    }
} AndOp;
} // namespace op::and::cuda

#endif // __AND_KERNEL_H__
