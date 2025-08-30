#ifndef __RELU_BACKWARD_CUDA_H__
#define __RELU_BACKWARD_CUDA_H__
// #include <cuda_runtime.h>
// #include <math.h>

namespace op::relu_backward::cuda {
typedef struct ReLUBackwardOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x, const T &grad_y) const {
        if (float(x) > 0)
            return grad_y;
        else
            return 0;
    }
} ReLUBackwardOp;
} // namespace op::relu_backward::cuda

#endif // __RELU_BACKWARD_KERNEL_H__
