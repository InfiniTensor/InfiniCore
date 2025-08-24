#ifndef GELU_CPU_H
#define GELU_CPU_H

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include <cmath>

ELEMENTWISE_DESCRIPTOR(gelu, cpu)

namespace op::gelu::cpu {
typedef struct GeluOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    T operator()(const T &x) const {
        // GELU approximation exactly matching PyTorch's implementation
        // PyTorch uses: 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

        // Use double precision for intermediate calculations to maintain accuracy
        double x_d = static_cast<double>(x);

        // Use exact same constants as PyTorch with higher precision
        const double sqrt_2_over_pi = 0.7978845608028654; // math.sqrt(2.0 / math.pi)
        const double alpha = 0.044715;

        // Follow PyTorch's exact computation order
        double x_pow_3 = x_d * x_d * x_d;                                          // torch.pow(x, 3.0)
        double alpha_x_pow_3 = alpha * x_pow_3;                                    // 0.044715 * torch.pow(x, 3.0)
        double x_plus_alpha_x_pow_3 = x_d + alpha_x_pow_3;                         // x + 0.044715 * torch.pow(x, 3.0)
        double sqrt_2_over_pi_times_inner = sqrt_2_over_pi * x_plus_alpha_x_pow_3; // math.sqrt(2.0 / math.pi) * (...)
        double tanh_result = std::tanh(sqrt_2_over_pi_times_inner);                // torch.tanh(...)
        double one_plus_tanh = 1.0 + tanh_result;                                  // 1.0 + torch.tanh(...)
        double x_times_one_plus_tanh = x_d * one_plus_tanh;                        // x * (1.0 + torch.tanh(...))
        double result = 0.5 * x_times_one_plus_tanh;                               // 0.5 * x * (1.0 + torch.tanh(...))

        return static_cast<T>(result);
    }
} GeluOp;
} // namespace op::gelu::cpu

#endif // GELU_CPU_H
