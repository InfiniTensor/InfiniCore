#ifndef __TANH_CPU_H__
#define __TANH_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include <cmath>

ELEMENTWISE_DESCRIPTOR(tanh, cpu)

namespace op::tanh::cpu {
typedef struct TanhOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    T operator()(const T &x) const {
        return std::tanh(x);
    }
} TanhOp;
} // namespace op::tanh::cpu

#endif // __TANH_CPU_H__