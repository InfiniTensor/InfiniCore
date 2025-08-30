#ifndef __RELU_BACKWARD_CPU_H__
#define __RELU_BACKWARD_CPU_H__

#include <type_traits>

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "../../../../utils/custom_types.h"

ELEMENTWISE_DESCRIPTOR(relu_backward, cpu)

namespace op::relu_backward::cpu {
typedef struct ReluBackwardOp {
public:
    static constexpr size_t num_inputs = 2;

    template <typename T>
    T operator()(const T &input, const T &grad_output) const {
        return input > static_cast<T>(0) ? grad_output : static_cast<T>(0);
    }

    // Specialization for bf16_t to use double for intermediate calculations
    bf16_t operator()(const bf16_t &input, const bf16_t &grad_output) const {
        double input_val = _bf16_to_f32(input);
        double grad_output_val = _bf16_to_f32(grad_output);
        return _f32_to_bf16(static_cast<float>(input_val > 0.0 ? grad_output_val : 0.0));
    }
} ReluBackwardOp;
} // namespace op::relu_backward::cpu

#endif // __RELU_BACKWARD_CPU_H__