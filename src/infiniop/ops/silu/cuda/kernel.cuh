#ifndef __SILU_CUDA_H__
#define __SILU_CUDA_H__
// #include <cuda_runtime.h>
// #include <math.h>

namespace op::silu::cuda {
typedef struct SiluOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        return float(x) / (1 + std::exp(-float(x)));
    }
} SiluOp;
} // namespace op::silu::cuda

#endif // __SILU_KERNEL_H__
