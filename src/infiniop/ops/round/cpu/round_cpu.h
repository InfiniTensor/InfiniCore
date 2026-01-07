#ifndef __ROUND_CPU_H__
#define __ROUND_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include <cmath>

ELEMENTWISE_DESCRIPTOR(round, cpu)

namespace op::round::cpu {
typedef struct RoundOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &x) const {
        if constexpr (std::is_integral_v<T>) {
            return x;
        } else {
            return std::nearbyint(x);
        }
    }
} RoundOp;
} // namespace op::round::cpu

#endif // __ROUND_CPU_H__
