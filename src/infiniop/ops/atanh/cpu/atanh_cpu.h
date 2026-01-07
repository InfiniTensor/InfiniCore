#ifndef __ATANH_CPU_H__
#define __ATANH_CPU_H__

#include <cmath>

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(atanh, cpu)

namespace op::atanh::cpu {
typedef struct AtanhOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &x) const {
        return std::atanh(x);
    }
} AtanhOp;
} // namespace op::atanh::cpu

#endif // __ATANH_CPU_H__
