#ifndef __POW_CPU_H__
#define __POW_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(pow, cpu)

namespace op::pow::cpu {
typedef struct PowOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    T operator()(const T &a, const T &b) const {
        return std::pow(a, b);
    }
} PowOp;
} // namespace op::pow::cpu

#endif // __POW_CPU_H__
