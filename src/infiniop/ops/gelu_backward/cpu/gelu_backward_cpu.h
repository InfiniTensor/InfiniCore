#ifndef __GELU_BACKWARD_CPU_H__
#define __GELU_BACKWARD_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(gelu_backward, cpu)

namespace op::gelu_backward::cpu {
typedef struct GeluBackwardOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    T operator()(const T &a, const T &b) const {
        constexpr double alpha = 0.7978845608028654;
        constexpr double beta = 0.044714998453855515;

        // 计算中间变量 u = α(x + βx³)
        const double x_cubed = a * a * a;
        const double u = alpha * (a + beta * x_cubed);

        // 计算 tanh(u) 及其导数 sech²(u) = 1 - tanh²(u)
        const double tanh_u = std::tanh(u);
        const double sech2_u = 1.0 - tanh_u * tanh_u;

        // 计算 du/dx = α(1 + 3βx²)
        const double du_dx = alpha * (1.0 + 3.0 * beta * a * a);

        // 计算 GELU 的导数 dy/dx
        const double dy_dx = 0.5 * (1.0 + tanh_u) + 0.5 * a * sech2_u * du_dx;

        // 链式法则：dL/dx = dL/dy * dy/dx
        const double ans = static_cast<double>(b) * dy_dx;
        return static_cast<T>(ans);
    }
} GeluBackwardOp;
} // namespace op::gelu_backward::cpu

#endif // __GELU_BACKWARD_CPU_H__
