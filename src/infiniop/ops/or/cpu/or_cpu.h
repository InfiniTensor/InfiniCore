#ifndef __OR_CPU_H__
#define __OR_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(or_op, cpu)

namespace op::or_op::cpu {
typedef struct OrOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    T operator()(const T &a, const T &b) const {
        // Logical OR: non-zero values are treated as true
        if constexpr (std::is_same_v<T, bool>) {
            return a || b;
        } else {
            return static_cast<T>((a != static_cast<T>(0)) || (b != static_cast<T>(0)));
        }
    }
} OrOp;

} // namespace op::or_op::cpu

#endif // __OR_CPU_H__