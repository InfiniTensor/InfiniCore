#ifndef __CEIL_CPU_H__
#define __CEIL_CPU_H__

#include <cmath>

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(ceil, cpu)

namespace op::ceil::cpu {
typedef struct CeilOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &x) const {
        if constexpr (std::is_integral_v<T>) {
            return x;
        } else {
            return std::ceil(x);
        }
    }
} CeilOp;
} // namespace op::ceil::cpu

#endif // __CEIL_CPU_H__
