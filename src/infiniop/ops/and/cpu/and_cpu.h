#ifndef __AND_CPU_H__
#define __AND_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
ELEMENTWISE_DESCRIPTOR(op_and, cpu)

namespace op::op_and::cpu {
typedef struct AndOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    T operator()(const T &a, const T &b) const {
        if constexpr (std::is_same<T, bool>::value)
            return a && b;
        else
            return utils::cast<bool>(a) && utils::cast<bool>(b);
    }
} AndOp;
} // namespace op::op_and::cpu


#endif // __AND_CPU_H__
