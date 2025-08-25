#ifndef TANH_CPU_H
#define TANH_CPU_H

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include <cmath>

ELEMENTWISE_DESCRIPTOR(tanh, cpu)

namespace op::tanh::cpu {
typedef struct TanhOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    T operator()(const T &x) const {
        // tanh(x) = (exp(2*x) - 1) / (exp(2*x) + 1)
        // or more stable: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        if constexpr (std::is_same_v<T, fp16_t>) {
            float x_f = static_cast<float>(x);
            return static_cast<T>(std::tanh(x_f));
        } else if constexpr (std::is_same_v<T, bf16_t>) {
            float x_f = static_cast<float>(x);
            return static_cast<T>(std::tanh(x_f));
        } else if constexpr (std::is_same_v<T, float>) {
            return std::tanh(x);
        } else if constexpr (std::is_same_v<T, double>) {
            return std::tanh(x);
        } else {
            return std::tanh(x);
        }
    }
} TanhOp;
} // namespace op::tanh::cpu

#endif // TANH_CPU_H