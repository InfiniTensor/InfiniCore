#ifndef __HARDSWISH_CPU_H__
#define __HARDSWISH_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include <algorithm>

ELEMENTWISE_DESCRIPTOR(hardswish, cpu)

namespace op::hardswish::cpu {
typedef struct HardswishOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &input) const {
        if constexpr (std::is_integral_v<T>) {
            return static_cast<T>(0);
        } else {
            // x * clamp(x + 3, 0, 6) / 6
            auto x = static_cast<double>(input);
            double y = x + 3.0;
            y = std::min(std::max(y, 0.0), 6.0);
            double out = x * (y / 6.0);
            return static_cast<T>(out);
        }
    }
} HardswishOp;
} // namespace op::hardswish::cpu

#endif // __HARDSWISH_CPU_H__
