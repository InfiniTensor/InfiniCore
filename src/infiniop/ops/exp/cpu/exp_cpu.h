#ifndef EXP_CPU_H
#define EXP_CPU_H

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include <cmath>

ELEMENTWISE_DESCRIPTOR(exp, cpu)

namespace op::exp::cpu {
typedef struct ExpOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    T operator()(const T &a) const {
        if constexpr (std::is_same_v<T, float>) {
            return std::exp(a);
        } else if constexpr (std::is_same_v<T, double>) {
            return std::exp(a);
        } else {
            // For fp16_t and bf16_t, convert to float, compute exp, then convert back
            return static_cast<T>(std::exp(static_cast<float>(a)));
        }
    }
} ExpOp;
} // namespace op::exp::cpu

#endif // EXP_CPU_H
