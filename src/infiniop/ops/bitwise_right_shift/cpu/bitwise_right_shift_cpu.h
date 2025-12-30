#ifndef __BITWISE_RIGHT_SHIFT_CPU_H__
#define __BITWISE_RIGHT_SHIFT_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(bitwise_right_shift, cpu)

namespace op::bitwise_right_shift::cpu {
typedef struct BitwiseRightShiftOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    T operator()(const T &x, const T &shift) const {
        return x >> shift;
    }
} BitwiseRightShiftOp;
} // namespace op::bitwise_right_shift::cpu

#endif // __BITWISE_RIGHT_SHIFT_CPU_H__
