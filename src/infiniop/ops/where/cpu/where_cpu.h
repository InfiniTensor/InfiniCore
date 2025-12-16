#ifndef __WHERE_CPU_H__
#define __WHERE_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

// Define Descriptor: op::where::cpu::Descriptor
ELEMENTWISE_DESCRIPTOR(where, cpu)

namespace op::where::cpu {

struct WhereOp {
public:
    // Three inputs: cond, x, y
    static constexpr size_t num_inputs = 3;

    // Homogeneous version: cond is already bool, x/y have same type as output
    template <typename T>
    T operator()(const bool &cond, const T &x, const T &y) const {
        return cond ? x : y;
    }

    // Heterogeneous version: support non-bool cond or explicit Tout
    template <typename Tout, typename Tcond, typename Tx, typename Ty>
    Tout operator()(const Tcond &cond, const Tx &x, const Ty &y) const {
        return static_cast<bool>(cond) ? static_cast<Tout>(x) : static_cast<Tout>(y);
    }
};

} // namespace op::where::cpu

#endif // __WHERE_CPU_H__
