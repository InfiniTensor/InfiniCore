#ifndef __SIGMOID_BACKWARD_CPU_H__
#define __SIGMOID_BACKWARD_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"


ELEMENTWISE_DESCRIPTOR(sigmoid_backward, cpu)

namespace op::sigmoid_backward::cpu {
typedef struct SigmoidBackwardOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    T operator()(const T &grad_output, const T &input) const {
        auto sigmoid = 1 / (1 + exp(-input));
        return sigmoid * (1 - sigmoid) * grad_output;
    }
} SigmoidBackwardOp;
}

#endif // __SIGMOID_BACKWARD_CPU_H__
