#ifndef __CROSSENTROPYLOSSBACKWARD_CUDA_H__
#define __CROSSENTROPYLOSSBACKWARD_CUDA_H__

#include <cstddef>
namespace op::CrossEntropyLossBackWard::cuda {
typedef struct CrossEntropyLossBackWardOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &probs, const T &target, const float N) const {
        // Cross Entropy Loss Backward: grad_logits = (probs - target) / batch_size
        if constexpr (std::is_same_v<T, half>) {
            return __hdiv(__hsub(probs, target), __float2half(N));
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            return __hdiv(__hsub(probs, target), __float2bfloat16(N));
        } else if constexpr (std::is_same_v<T, float>) {
            return __fdiv_rn(__fsub_rn(probs, target), N);
        } else {
            return (probs - target) / N;
        }
    }
} CrossEntropyLossBackWardOp;
} // namespace op::CrossEntropyLossBackWard::cuda

#endif // __CrossEntropyLossBackWard_CUDA_H__
