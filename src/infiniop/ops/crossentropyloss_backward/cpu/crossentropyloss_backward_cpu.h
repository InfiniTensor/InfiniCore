#ifndef __CrossEntropyLossBackWard_CPU_H__
#define __CrossEntropyLossBackWard_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(CrossEntropyLossBackWard, cpu)

namespace op::CrossEntropyLossBackWard::cpu {
typedef struct CrossEntropyLossBackWardOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    T operator()(const T &probs, const T &target, const size_t N) const {
        // Cross Entropy Loss Backward: grad_logits = (probs - target) / batch_size
        return (probs - target) / static_cast<T>(N);
    }
} CrossEntropyLossBackWardOp;
} // namespace op::CrossEntropyLossBackWard::cpu

#endif // __CrossEntropyLossBackWard_CPU_H__
