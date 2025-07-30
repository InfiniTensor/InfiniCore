#ifndef __GELU_BACKWARD_CPU_H__
#define __GELU_BACKWARD_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include <cmath>

ELEMENTWISE_DESCRIPTOR(gelu_backward, cpu)

namespace op::gelu_backward::cpu {
typedef struct GeluBackWardOp {
    float pi = 3.1415927f;
    float kappa = 0.044715f;
    float Beta = sqrt(2 / pi);

public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    T operator()(const T &input, const T &grad_output) const {
        // use Approximate formula Gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.0044715 * x ** 3 )))
        // \kappa = 0.044715, \beta = sqrt(2 / pi)
        //  inner() = \beta * (x + \kappa * x ** 3)
        //  SO: Gelu(x) = 0.5 * x * (1 + tanh(inner()))
        //
        // d(Gelu(x)) = 0.5 [(1 + tanh(inner())) + x(1 - tanh(inner) ** 2)(d(inner()))]
#define CREATE_D_DEUL(x)                           \
    float inner = Beta * (x + kappa * x * x * x);  \
    float dinner = Beta * (1 + 3 * kappa * x * x); \
    float tanh_inner = std::tanh(inner);           \
    float dGelu = 0.5 * ((1 + tanh_inner) + x * (1 - tanh_inner * tanh_inner) * dinner);

        if constexpr (std::is_same<T, fp16_t>::value) {
            float finput = _f16_to_f32(input);
            CREATE_D_DEUL(finput)

            return grad_output * _f32_to_f16(dGelu);

        } else if constexpr (std::is_same<T, bf16_t>::value) {
            float finput = _bf16_to_f32(input);
            CREATE_D_DEUL(finput)

            return grad_output * _f32_to_bf16(dGelu);

        } else if constexpr (std::is_same<T, float>::value) {
            CREATE_D_DEUL(input)

            return grad_output * dGelu;
        }
    }
} GeluBackWardOp;
} // namespace op::gelu_backward::cpu

#endif
