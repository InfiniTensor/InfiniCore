#ifndef __AND_CPU_H__
#define __AND_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(and_op, cpu)

namespace op::and_op::cpu {
typedef struct AndOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    T operator()(const T &a, const T &b) const {
        // Logical AND: non-zero values are treated as true
        if constexpr (std::is_same_v<T, bool>) {
            return a && b;
        } else {
            return static_cast<T>((a != static_cast<T>(0)) && (b != static_cast<T>(0)));
        }
    }
} AndOp;

} // namespace op::and_op::cpu

#endif // __AND_CPU_H__