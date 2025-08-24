#ifndef __HARD_SWISH_CPU_H__
#define __HARD_SWISH_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include <cmath>
#include <iostream>

ELEMENTWISE_DESCRIPTOR(hard_swish, cpu)

namespace op::hard_swish::cpu {
typedef struct HardSwishOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    T operator()(const T &a) const {
        T relu6 = std::min(std::max(a + static_cast<T>(3), static_cast<T>(0)), static_cast<T>(6));
        return a * relu6 / static_cast<T>(6);
    }
} HardSwishOp;
} // namespace op::hard_swish::cpu

#endif // __HARD_SWISH_CPU_H__
