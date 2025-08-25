#ifndef __WHERE_CPU_H__
#define __WHERE_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(where, cpu)

namespace op::where::cpu {
typedef struct WhereOp {
public:
    static constexpr size_t num_inputs = 3;
    template <typename T>
    T operator()(const T &a, const T &b, const bool &cond) const {
        return cond ? a : b;
    }
} WhereOp;
} // namespace op::where::cpu

#endif // __WHERE_CPU_H__
