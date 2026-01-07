#ifndef __ATAN_CPU_H__
#define __ATAN_CPU_H__

#include <cmath>

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(atan, cpu)

namespace op::atan::cpu {
typedef struct AtanOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &x) const {
        return std::atan(x);
    }
} AtanOp;
} // namespace op::atan::cpu

#endif // __ATAN_CPU_H__
