#ifndef __SQRT_CPU_H__
#define __SQRT_CPU_H__

#include <cmath>

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(sqrt, cpu)

namespace op::sqrt::cpu {
typedef struct SqrtOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &x) const {
        return std::sqrt(x);
    }
} SqrtOp;
} // namespace op::sqrt::cpu

#endif // __SQRT_CPU_H__
