#ifndef __GELU_BACKWARD_CPU_H__
#define __GELU_BACKWARD_CPU_H__

#include <cmath>
#include <type_traits>

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "../../../../utils/custom_types.h"

ELEMENTWISE_DESCRIPTOR(gelu_backward, cpu)

namespace op::gelu_backward::cpu {
typedef struct GeluBackwardOp {
public:
    static constexpr size_t num_inputs = 2;

    template <typename T>
    T operator()(const T &grad_output, const T &input) const {
        // GeLU derivative using tanh approximation
        // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        // d/dx GELU(x) ≈ 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³))) + 
        //                0.5 * x * (1 - tanh²(sqrt(2/π) * (x + 0.044715 * x³))) * sqrt(2/π) * (1 + 3 * 0.044715 * x²)
        
        constexpr T sqrt_2_over_pi = static_cast<T>(0.7978845608028654);
        constexpr T coeff = static_cast<T>(0.044715);
        
        T x = input;
        T x_cubed = x * x * x;
        T inner = sqrt_2_over_pi * (x + coeff * x_cubed);
        T tanh_val = std::tanh(inner);
        T tanh_squared = tanh_val * tanh_val;
        
        T term1 = static_cast<T>(0.5) * (static_cast<T>(1.0) + tanh_val);
        T term2 = static_cast<T>(0.5) * x * (static_cast<T>(1.0) - tanh_squared) * sqrt_2_over_pi * (static_cast<T>(1.0) + static_cast<T>(3.0) * coeff * x * x);
        
        T gelu_derivative = term1 + term2;
        
        return grad_output * gelu_derivative;
    }

    // Specialization for float to use double for intermediate calculations
    float operator()(const float &grad_output, const float &input) const {
        double x = static_cast<double>(input);
        double grad_out = static_cast<double>(grad_output);
        
        constexpr double sqrt_2_over_pi = 0.7978845608028654;
        constexpr double coeff = 0.044715;
        
        double x_cubed = x * x * x;
        double inner = sqrt_2_over_pi * (x + coeff * x_cubed);
        double tanh_val = std::tanh(inner);
        double tanh_squared = tanh_val * tanh_val;
        
        double term1 = 0.5 * (1.0 + tanh_val);
        double term2 = 0.5 * x * (1.0 - tanh_squared) * sqrt_2_over_pi * (1.0 + 3.0 * coeff * x * x);
        
        double gelu_derivative = term1 + term2;
        
        return static_cast<float>(grad_out * gelu_derivative);
    }

    // Specialization for bf16_t to use double for intermediate calculations
    bf16_t operator()(const bf16_t &grad_output, const bf16_t &input) const {
        double x = _bf16_to_f32(input);
        double grad_out = _bf16_to_f32(grad_output);
        
        constexpr double sqrt_2_over_pi = 0.7978845608028654;
        constexpr double coeff = 0.044715;
        
        double x_cubed = x * x * x;
        double inner = sqrt_2_over_pi * (x + coeff * x_cubed);
        double tanh_val = std::tanh(inner);
        double tanh_squared = tanh_val * tanh_val;
        
        double term1 = 0.5 * (1.0 + tanh_val);
        double term2 = 0.5 * x * (1.0 - tanh_squared) * sqrt_2_over_pi * (1.0 + 3.0 * coeff * x * x);
        
        double gelu_derivative = term1 + term2;
        
        return _f32_to_bf16(static_cast<float>(grad_out * gelu_derivative));
    }
} GeluBackwardOp;
} // namespace op::gelu_backward::cpu

#endif // __GELU_BACKWARD_CPU_H__