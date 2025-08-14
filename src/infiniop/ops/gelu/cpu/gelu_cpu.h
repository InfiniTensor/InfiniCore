#ifndef __GELU_CPU_H__
#define __GELU_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(gelu, cpu)

#include <cmath>

namespace op::gelu::cpu {
typedef struct GeluOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &x) const {
        constexpr double Alpha = 0.7978845608028654;
        constexpr double Beta = 0.044715;
        double inner = x + Beta * x * x * x;
        double tanh_term = std::tanh(Alpha * inner);
        return static_cast<T>(0.5 * x * (1.0 + tanh_term));
    }
} GeluOp;

} // namespace op::gelu::cpu

#endif // __GELU_CPU_H__
