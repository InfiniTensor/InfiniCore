#ifndef __CROSS_ENTROPY_LOSS_BACKWARD_CUDA_H__
#define __CROSS_ENTROPY_LOSS_BACKWARD_CUDA_H__
// #include <cuda_runtime.h>
// #include <math.h>

namespace op::cross_entropy_loss_backward::cuda {
typedef struct CrossEntropyLossBackwardOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &probs, const T &target, T batch_size) const {
        return (float(probs) - float(target)) / float(batch_size);
    }
} CrossEntropyLossBackwardOp;
} // namespace op::cross_entropy_loss_backward::cuda

#endif // __CROSS_ENTROPY_LOSS_BACKWARD_KERNEL_H__
