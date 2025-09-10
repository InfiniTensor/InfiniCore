#ifndef COS_CPU_H
#define COS_CPU_H

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include <cmath>

ELEMENTWISE_DESCRIPTOR(cos, cpu)

namespace op::cos::cpu {
typedef struct CosOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    T operator()(const T &x) const {
        // cos(x) = cosine of x
        if constexpr (std::is_same_v<T, fp16_t>) {
            float x_f = static_cast<float>(x);
            return static_cast<T>(std::cos(x_f));
        } else if constexpr (std::is_same_v<T, bf16_t>) {
            float x_f = static_cast<float>(x);
            return static_cast<T>(std::cos(x_f));
        } else if constexpr (std::is_same_v<T, float>) {
            return std::cos(x);
        } else if constexpr (std::is_same_v<T, double>) {
            return std::cos(x);
        } else {
            return std::cos(x);
        }
    }
} CosOp;
} // namespace op::cos::cpu

#endif // COS_CPU_H