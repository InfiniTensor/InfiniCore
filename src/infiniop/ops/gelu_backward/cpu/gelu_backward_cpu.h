#ifndef __GELU_BACKWARD_CPU_H__
#define __GELU_BACKWARD_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include <cmath>

ELEMENTWISE_DESCRIPTOR(gelu_backward, cpu)

namespace op::gelu_backward::cpu {
typedef struct GeluBackwardOp {
public:
    static constexpr size_t num_inputs = 2;

    template <typename T>
    T operator()(const T &input, const T &grad_output) const {
        // GeLU backward using approximation: 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3))) + x * derivative_part
        constexpr T sqrt_2_over_pi = static_cast<T>(0.7978845608028654); // sqrt(2/π)
        constexpr T coeff = static_cast<T>(0.044715);
        constexpr T half = static_cast<T>(0.5);
        constexpr T one = static_cast<T>(1.0);

        T x = input;
        T x_cubed = x * x * x;
        T tanh_arg = sqrt_2_over_pi * (x + coeff * x_cubed);
        T tanh_val = std::tanh(tanh_arg);

        // Derivative of tanh
        T sech_squared = one - tanh_val * tanh_val; // sech^2 = 1 - tanh^2

        // Derivative of the argument inside tanh
        T arg_derivative = sqrt_2_over_pi * (one + static_cast<T>(3.0) * coeff * x * x);

        // Complete derivative of GeLU
        T gelu_derivative = half * (one + tanh_val) + x * half * sech_squared * arg_derivative;

        return grad_output * gelu_derivative;
    }
} GeluBackwardOp;
} // namespace op::gelu_backward::cpu

#endif // __GELU_BACKWARD_CPU_H__
