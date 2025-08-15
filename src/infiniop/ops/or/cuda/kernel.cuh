#ifndef __OR_CUDA_H__
#define __OR_CUDA_H__
// #include <cuda_runtime.h>
// #include <math.h>

namespace op::op_or::cuda {
typedef struct OrOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        if constexpr (std::is_same<T, bool>::value)
            return a || b;
        else
            return bool(a) || bool(b);
    }
} OrOp;
} // namespace op::op_or::cuda

#endif // __OR_KERNEL_H__
