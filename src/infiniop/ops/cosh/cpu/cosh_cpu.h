#ifndef __COSH_CPU_H__
#define __COSH_CPU_H__

#include <cmath>

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(cosh, cpu)

namespace op::cosh::cpu {
typedef struct CoshOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &x) const {
        return std::cosh(x);
    }
} CoshOp;
} // namespace op::cosh::cpu

#endif // __COSH_CPU_H__
