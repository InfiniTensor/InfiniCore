#ifndef __CROSS_ENTROPY_LOSS_BACKWARD_CPU_H__
#define __CROSS_ENTROPY_LOSS_BACKWARD_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(cross_entropy_loss_backward, cpu)

namespace op::cross_entropy_loss_backward::cpu {
typedef struct CrossEntropyLossBackwardOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    T operator()(const T &a, const T &b, const size_t N) const {
        return (a - b) / static_cast<T>(N);
    }
} CrossEntropyLossBackwardOp;
} // namespace op::cross_entropy_loss_backward::cpu

#endif // __CROSS_ENTROPY_LOSS_BACKWARD_CPU_H__
