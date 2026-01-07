#ifndef __FLOOR_CPU_H__
#define __FLOOR_CPU_H__

#include <cmath>

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(floor, cpu)

namespace op::floor::cpu {
typedef struct FloorOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &x) const {
        if constexpr (std::is_integral_v<T>) {
            return x;
        } else {
            return std::floor(x);
        }
    }
} FloorOp;
} // namespace op::floor::cpu

#endif // __FLOOR_CPU_H__
