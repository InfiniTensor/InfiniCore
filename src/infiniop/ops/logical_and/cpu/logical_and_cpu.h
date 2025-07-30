#ifndef __LOGICAL_AND_CPU_H__
#define __LOGICAL_AND_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(logical_and, cpu)

namespace op::logical_and::cpu {
typedef struct LogicalAndOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    T operator()(const T &a, const T &b) const {
        return a && b;
    }
} LogicalAndOp;
} // namespace op::logical_and::cpu

#endif
