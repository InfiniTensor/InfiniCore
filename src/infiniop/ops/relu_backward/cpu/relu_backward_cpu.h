#ifndef __RELU_BACKWARD_CPU_H__
#define __RELU_BACKWARD_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(relu_backward, cpu)

namespace op::relu_backward::cpu {
typedef struct ReluBackwardOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    T operator()(const T &a, const T &b) const {
        T zero{0};
        if (a > zero) {
            return b;
        } else {
            return zero;
        }
    }
} ReluBackwardOp;
} // namespace op::relu_backward::cpu

#endif // __RELU_BACKWARD_CPU_H__
