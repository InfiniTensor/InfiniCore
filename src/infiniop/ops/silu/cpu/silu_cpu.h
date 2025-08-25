#ifndef __SILU_CPU_H__
#define __SILU_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(silu, cpu)

namespace op::silu::cpu {
typedef struct SiluOp {
private:
    template <typename T>
    T sigmoid(const T &x) const {
        return T(1) / (T(1) + std::exp(-x));
    }

public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    T operator()(const T &x) const {
        return x * sigmoid(x);
    }
} SiluOp;
} // namespace op::silu::cpu

#endif
