#ifndef __GELU_CUDA_H__
#define __GELU_CUDA_H__
// #include <cuda_runtime.h>
// #include <math.h>

namespace op::gelu::cuda {
typedef struct GeLUOp {
public:
    static constexpr size_t num_inputs = 1;
    const float sqrt_2_over_pi = sqrtf(2 / 3.14159f);
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        float x_f = float(x);
        return 0.5 * x_f * (1 + tanhf(sqrt_2_over_pi * (x_f + 0.044715f * x_f * x_f * x_f)));
    }
} GeLUOp;
} // namespace op::gelu::cuda

#endif // __GELU_KERNEL_H__
