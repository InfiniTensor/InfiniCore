#ifndef __LOGICAL_OR_CPU_H__
#define __LOGICAL_OR_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(logical_or, cpu)

namespace op::logical_or::cpu {
typedef struct LogicalOrOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    T operator()(const T &a, const T &b) const {
        return a || b;
    }
} LogicalOrOp;
} // namespace op::logical_or::cpu

#endif
