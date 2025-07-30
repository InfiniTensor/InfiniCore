#ifndef __GELU_CPU_H__
#define __GELU_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include <cmath>

ELEMENTWISE_DESCRIPTOR(gelu, cpu)

namespace op::gelu::cpu {
typedef struct GeluOp {
    float pi = 3.1415927f;

    float kappa = 0.044715;
    float beta = sqrt(2 / pi);

public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    T operator()(const T &input) const {
        // use Approximate formula Gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.0044715 * x **3)))
        // \kappa = 0.044715, \beta = sqrt(2 / pi)
        //  inner() = \beta * (x + \kappa * x ** 3)
        //  SO: Gelu(x) = 0.5 * x * (1 + tanh(inner()))
        //
        if constexpr (std::is_same<T, fp16_t>::value) {
            float finput = _f16_to_f32(input);
            float inner = beta * (finput + kappa * finput * finput * finput);
            float res = 0.5 * finput * (1 + std::tanh(inner));

            return _f32_to_f16(res);
        } else if constexpr (std::is_same<T, bf16_t>::value) {
            float finput = _bf16_to_f32(input);
            float inner = beta * (finput + kappa * finput * finput * finput);
            float res = 0.5 * finput * (1 + std::tanh(inner));

            return _f32_to_bf16(res);
        } else if constexpr (std::is_same<T, float>::value) {
            float inner = beta * (input + kappa * input * input * input);
            return 0.5 * input * (1 + std::tanh(inner));

        } else {
            float inner = beta * (input + kappa * input * input * input);
            return 0.5 * input * (1 + std::tanh(inner));
        }
    }
} GeluOp;
} // namespace op::gelu::cpu

#endif
