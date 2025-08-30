#ifndef __OR_CPU_H__
#define __OR_CPU_H__

#include <type_traits>

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "../../../../utils/custom_types.h"

ELEMENTWISE_DESCRIPTOR(or_op, cpu)

namespace op::or_op::cpu {
typedef struct OrOp {
public:
    static constexpr size_t num_inputs = 2;

    bool operator()(const bool &a, const bool &b) const {
        return a || b;
    }
} OrOp;
} // namespace op::or_op::cpu

#endif // __OR_CPU_H__