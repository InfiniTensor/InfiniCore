#ifndef RELU_BACKWARD_CPU_H
#define RELU_BACKWARD_CPU_H

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(relu_backward, cpu)

namespace op::relu_backward::cpu {
typedef struct ReluBackwardOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    T operator()(const T &input, const T &grad_output) const {
        return input > T(0) ? grad_output : T(0);
    }
} ReluBackwardOp;

} // namespace op::relu_backward::cpu

#endif // RELU_BACKWARD_CPU_H