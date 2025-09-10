#ifndef SIGMOID_BACKWARD_CPU_H
#define SIGMOID_BACKWARD_CPU_H

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include <cmath>

ELEMENTWISE_DESCRIPTOR(sigmoid_backward, cpu)

namespace op::sigmoid_backward::cpu {
typedef struct SigmoidBackwardOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    T operator()(const T &input, const T &grad_output) const {
        // sigmoid_backward: grad_input = grad_output * sigmoid(input) * (1 - sigmoid(input))
        T sigmoid_val;
        if constexpr (std::is_same_v<T, fp16_t>) {
            sigmoid_val = static_cast<T>(1.0f / (1.0f + std::exp(-static_cast<float>(input))));
        } else if constexpr (std::is_same_v<T, bf16_t>) {
            sigmoid_val = static_cast<T>(1.0f / (1.0f + std::exp(-static_cast<float>(input))));
        } else {
            sigmoid_val = static_cast<T>(1.0) / (static_cast<T>(1.0) + std::exp(-input));
        }
        return grad_output * sigmoid_val * (static_cast<T>(1.0) - sigmoid_val);
    }
} SigmoidBackwardOp;
} // namespace op::sigmoid_backward::cpu

#endif // SIGMOID_BACKWARD_CPU_H
