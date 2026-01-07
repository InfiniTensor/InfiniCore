#ifndef __ACOSH_CPU_H__
#define __ACOSH_CPU_H__

#include <cmath>

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(acosh, cpu)

namespace op::acosh::cpu {
typedef struct AcoshOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &x) const {
        return std::acosh(x);
    }
} AcoshOp;
} // namespace op::acosh::cpu

#endif // __ACOSH_CPU_H__
