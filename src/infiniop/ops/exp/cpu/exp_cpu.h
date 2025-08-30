#ifndef __EXP_CPU_H__
#define __EXP_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include <cmath>

ELEMENTWISE_DESCRIPTOR(exp, cpu)

namespace op::exp::cpu {
typedef struct ExpOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &input) const {
        return std::exp(input);
    }
} ExpOp;
} // namespace op::exp::cpu

#endif // __EXP_CPU_H__
