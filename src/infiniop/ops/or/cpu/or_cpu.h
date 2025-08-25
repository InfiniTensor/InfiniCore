#ifndef __OR_CPU_H__
#define __OR_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
ELEMENTWISE_DESCRIPTOR(op_or, cpu)

namespace op::op_or::cpu {
typedef struct OrOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    T operator()(const T &a, const T &b) const {
        if constexpr (std::is_same<T, bool>::value)
            return a || b;
        else
            return utils::cast<bool>(a) || utils::cast<bool>(b);
    }
} OrOp;
} // namespace op::op_or::cpu


#endif // __OR_CPU_H__
