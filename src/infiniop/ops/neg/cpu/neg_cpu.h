#ifndef __NEG_CPU_H__
#define __NEG_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(neg, cpu)

namespace op::neg::cpu {
typedef struct NegOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &x) const {
        return -x;
    }
} NegOp;
} // namespace op::neg::cpu

#endif // __NEG_CPU_H__
