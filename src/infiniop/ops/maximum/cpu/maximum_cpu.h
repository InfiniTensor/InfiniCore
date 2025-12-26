#ifndef __MAXIMUM_CPU_H__
#define __MAXIMUM_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(maximum, cpu)

namespace op::maximum::cpu {
typedef struct MaximumOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    T operator()(const T &a, const T &b) const {
        return a > b ? a : b;
    }
} MaximumOp;
} // namespace op::maximum::cpu

#endif // __MAXIMUM_CPU_H__
