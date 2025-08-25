#ifndef __GELU_CPU_H__
#define __GELU_CPU_H__

#include <cmath>
#include <type_traits>

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "../../../../utils/custom_types.h"

ELEMENTWISE_DESCRIPTOR(gelu, cpu)

namespace op::gelu::cpu {
typedef struct GeluOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &x) const {
        // GeLU tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        constexpr T sqrt_2_over_pi = static_cast<T>(0.7978845608028654);
        constexpr T coeff = static_cast<T>(0.044715);
        T x_cubed = x * x * x;
        T tanh_input = sqrt_2_over_pi * (x + coeff * x_cubed);
        return static_cast<T>(0.5) * x * (static_cast<T>(1.0) + std::tanh(tanh_input));
    }

    // Specialization for float to use double for intermediate calculations
    float operator()(const float &x) const {
        double x_val = static_cast<double>(x);
        constexpr double sqrt_2_over_pi = 0.7978845608028654;
        constexpr double coeff = 0.044715;
        double x_cubed = x_val * x_val * x_val;
        double tanh_input = sqrt_2_over_pi * (x_val + coeff * x_cubed);
        double result = 0.5 * x_val * (1.0 + std::tanh(tanh_input));
        return static_cast<float>(result);
    }

    // Specialization for bf16_t to use double for intermediate calculations
    bf16_t operator()(const bf16_t &x) const {
        double x_val = _bf16_to_f32(x);
        constexpr double sqrt_2_over_pi = 0.7978845608028654;
        constexpr double coeff = 0.044715;
        double x_cubed = x_val * x_val * x_val;
        double tanh_input = sqrt_2_over_pi * (x_val + coeff * x_cubed);
        double result = 0.5 * x_val * (1.0 + std::tanh(tanh_input));
        return _f32_to_bf16(static_cast<float>(result));
    }
} GeluOp;
} // namespace op::gelu::cpu

#endif // __GELU_CPU_H__