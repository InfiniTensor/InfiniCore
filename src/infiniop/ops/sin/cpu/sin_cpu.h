#ifndef SIN_CPU_H
#define SIN_CPU_H

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include <cmath>

ELEMENTWISE_DESCRIPTOR(sin, cpu)

namespace op::sin::cpu {
typedef struct SinOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    T operator()(const T &x) const {
        // sin(x)
        if constexpr (std::is_same_v<T, fp16_t>) {
            float x_f = static_cast<float>(x);
            return static_cast<T>(std::sin(x_f));
        } else if constexpr (std::is_same_v<T, bf16_t>) {
            float x_f = static_cast<float>(x);
            return static_cast<T>(std::sin(x_f));
        } else if constexpr (std::is_same_v<T, float>) {
            return std::sin(x);
        } else if constexpr (std::is_same_v<T, double>) {
            return std::sin(x);
        } else {
            return std::sin(x);
        }
    }
} SinOp;
} // namespace op::sin::cpu

#endif // SIN_CPU_H