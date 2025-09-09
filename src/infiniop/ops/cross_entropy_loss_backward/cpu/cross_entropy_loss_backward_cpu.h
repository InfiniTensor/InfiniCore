#ifndef CROSS_ENTROPY_LOSS_BACKWARD_CPU_H
#define CROSS_ENTROPY_LOSS_BACKWARD_CPU_H

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(cross_entropy_loss_backward, cpu)

namespace op::cross_entropy_loss_backward::cpu {

struct CrossEntropyLossBackwardOp {
    static constexpr size_t num_inputs = 2;

    template <typename T>
    T operator()(const T &probs, const T &target, const size_t batch_size) const {
        // grad_logits = (probs - target) / batch_size  (reduction='mean')
        T scale = static_cast<T>(1.0) / static_cast<T>(batch_size);
        return (probs - target) * scale;
    }
};

} // namespace op::cross_entropy_loss_backward::cpu

#endif // CROSS_ENTROPY_LOSS_BACKWARD_CPU_H
