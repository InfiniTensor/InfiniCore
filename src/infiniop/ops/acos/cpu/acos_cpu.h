#ifndef __ACOS_CPU_H__
#define __ACOS_CPU_H__

#include <cmath>

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(acos, cpu)

namespace op::acos::cpu {
typedef struct AcosOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &x) const {
        return std::acos(x);
    }
} AcosOp;
} // namespace op::acos::cpu

#endif // __ACOS_CPU_H__
