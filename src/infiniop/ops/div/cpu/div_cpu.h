#ifndef __DIV_CPU_H__
#define __DIV_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(div, cpu)

namespace op::div::cpu {
typedef struct DivOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    T operator()(const T &a, const T &b) const {
        return a / b;
    }
} DivOp;
} // namespace op::div::cpu

#endif // __DIV_CPU_H__
