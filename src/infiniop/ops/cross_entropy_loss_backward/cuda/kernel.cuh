#ifndef CROSS_ENTROPY_LOSS_BACKWARD_CUDA_H
#define CROSS_ENTROPY_LOSS_BACKWARD_CUDA_H

namespace op::cross_entropy_loss_backward::cuda {
typedef struct CrossEntropyLossBackwardOp {
public:
    static constexpr size_t num_inputs = 2;
    // CrossEntropyLossBackwardOp(size_t batch_size) : batch_size_(batch_size) {}

    template <typename T>
    __device__ __forceinline__ T operator()(const T &probs, const T &target, const size_t batch_size) const {
        // grad_logits = (probs - target) / batch_size (reduction='mean')
        T diff;
        T scale;

        if constexpr (std::is_same_v<T, half>) {
            diff = __hsub(probs, target);
            scale = static_cast<half>(1.0) / __float2half(static_cast<float>(batch_size));
            return __hmul(diff, scale);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            diff = __hsub(probs, target);
            scale = static_cast<cuda_bfloat16>(1.0) / __float2bfloat16(static_cast<float>(batch_size));
            return __hmul(diff, scale);
        } else if constexpr (std::is_same_v<T, float>) {
            diff = __fsub_rd(probs, target);
            scale = 1.0 / batch_size;
            return __fmul_rd(diff, scale);
        } else {
            // fallback for other types (double, etc.)
            diff = probs - target;
            scale = 1.0 / batch_size;
            return diff * scale;
        }
    }

} CrossEntropyLossBackwardOp;
} // namespace op::cross_entropy_loss_backward::cuda

#endif // CROSS_ENTROPY_LOSS_BACKWARD_CUDA_H