
#ifndef HARDSWISH_CPU_H
#define HARDSWISH_CPU_H

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include <algorithm>
#include <cmath>

ELEMENTWISE_DESCRIPTOR(hardswish, cpu)

namespace op::hardswish::cpu {
typedef struct HardSwishOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    T operator()(const T &x) const {
        // HardSwish(x) = x * HardSigmoid(x) = x * max(0, min(1, (x + 3) / 6))
        if constexpr (std::is_same_v<T, fp16_t>) {
            float x_f = static_cast<float>(x);
            float hard_sigmoid = std::max(0.0f, std::min(1.0f, (x_f + 3.0f) / 6.0f));
            return static_cast<T>(x_f * hard_sigmoid);
        } else if constexpr (std::is_same_v<T, bf16_t>) {
            float x_f = static_cast<float>(x);
            float hard_sigmoid = std::max(0.0f, std::min(1.0f, (x_f + 3.0f) / 6.0f));
            return static_cast<T>(x_f * hard_sigmoid);
        } else if constexpr (std::is_same_v<T, float>) {
            float hard_sigmoid = std::max(0.0f, std::min(1.0f, (x + 3.0f) / 6.0f));
            return x * hard_sigmoid;
        } else if constexpr (std::is_same_v<T, double>) {
            double hard_sigmoid = std::max(0.0, std::min(1.0, (x + 3.0) / 6.0));
            return x * hard_sigmoid;
        } else {
            float x_f = static_cast<float>(x);
            float hard_sigmoid = std::max(0.0f, std::min(1.0f, (x_f + 3.0f) / 6.0f));
            return static_cast<T>(x_f * hard_sigmoid);
        }
    }
} HardSwishOp;
} // namespace op::hardswish::cpu

#endif // HARDSWISH_CPU_H