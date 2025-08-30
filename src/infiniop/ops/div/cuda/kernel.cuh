#ifndef __DIV_CUDA_H__
#define __DIV_CUDA_H__
// #include <cuda_runtime.h>
// #include <math.h>

namespace op::div::cuda {
typedef struct DivOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return float(a) / float(b);
    }
} DivOp;
} // namespace op::div::cuda

#endif // __DIV_KERNEL_H__
