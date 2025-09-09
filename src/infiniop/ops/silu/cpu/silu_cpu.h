#ifndef SILU_CPU_H
#define SILU_CPU_H

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include <cmath>

ELEMENTWISE_DESCRIPTOR(silu, cpu)

namespace op::silu::cpu {
typedef struct SiluOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    T operator()(const T &x) const {
        // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
        if constexpr (std::is_same_v<T, fp16_t>) {
            float x_f = static_cast<float>(x);
            return static_cast<T>(x_f / (1.0f + std::exp(-x_f)));
        } else if constexpr (std::is_same_v<T, bf16_t>) {
            float x_f = static_cast<float>(x);
            return static_cast<T>(x_f / (1.0f + std::exp(-x_f)));
        } else if constexpr (std::is_same_v<T, float>) {
            return x / (1.0f + std::exp(-x));
        } else if constexpr (std::is_same_v<T, double>) {
            return x / (1.0 + std::exp(-x));
        } else {
            return x / (1.0f + std::exp(-x));
        }
    }
} SiluOp;
} // namespace op::silu::cpu

#endif // SILU_CPU_H