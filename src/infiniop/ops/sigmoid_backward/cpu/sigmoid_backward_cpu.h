#ifndef __SIGMOID_BACKWARD_CPU_H__
#define __SIGMOID_BACKWARD_CPU_H__

#include <cmath>
#include <type_traits>

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "../../../../utils/custom_types.h"

ELEMENTWISE_DESCRIPTOR(sigmoid_backward, cpu)

namespace op::sigmoid_backward::cpu {
typedef struct SigmoidBackwardOp {
public:
    static constexpr size_t num_inputs = 2;

    template <typename T>
    T operator()(const T &input, const T &grad_output) const {
        // Sigmoid backward: grad_input = grad_output * sigmoid(input) * (1 - sigmoid(input))
        T sigmoid_val = T(1) / (T(1) + std::exp(-input));
        return grad_output * sigmoid_val * (T(1) - sigmoid_val);
    }
    
    // 为bf16类型特化，使用double作为中间计算类型以提高精度
    bf16_t operator()(const bf16_t &input, const bf16_t &grad_output) const {
        // 将bf16转换为double进行计算，然后再转回bf16
        double input_double = static_cast<double>(_bf16_to_f32(input));
        double grad_output_double = static_cast<double>(_bf16_to_f32(grad_output));
        
        // Sigmoid backward计算
        double sigmoid_val = 1.0 / (1.0 + std::exp(-input_double));
        double result = grad_output_double * sigmoid_val * (1.0 - sigmoid_val);
        
        // 使用utils::cast从double直接转换到bf16，保留更高精度
        return utils::cast<bf16_t>(result);
    }
} SigmoidBackwardOp;
} // namespace op::sigmoid_backward::cpu

#endif // __SIGMOID_BACKWARD_CPU_H__