#ifndef __LOGICAL_EQUAL_CPU_H__
#define __LOGICAL_EQUAL_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(logical_equal, cpu)

namespace op::logical_equal::cpu {
typedef struct LogicalEqualOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    T operator()(const T &a, const T &b) const {
        return a == b;
    }
} LogicalEqualOp;
} // namespace op::logical_equal::cpu

#endif
