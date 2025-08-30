#ifndef __AND_CPU_H__
#define __AND_CPU_H__

#include <type_traits>

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "../../../../utils/custom_types.h"

ELEMENTWISE_DESCRIPTOR(and_op, cpu)

namespace op::and_op::cpu {
typedef struct AndOp {
public:
    static constexpr size_t num_inputs = 2;

    bool operator()(const bool &a, const bool &b) const {
        return a && b;
    }
} AndOp;
} // namespace op::and_op::cpu

#endif // __AND_CPU_H__