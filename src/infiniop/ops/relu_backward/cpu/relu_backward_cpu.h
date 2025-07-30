#ifndef __RELU_BACKWARD_CPU_H__
#define __RELU_BACKWARD_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(relu_backward, cpu)

namespace op::relu_backward::cpu {
typedef struct ReluOp {
public:
    static constexpr size_t num_inputs = 2;

    template <typename T>
    T operator()(const T &input, const T &grad_output) const {
        auto dy = T(1) ? input > T(0) : T(0);
        return grad_output * dy;
    }
} ReluOp;
} // namespace op::relu_backward::cpu

#endif // __RELU_BACKWARD_CPU_H__
