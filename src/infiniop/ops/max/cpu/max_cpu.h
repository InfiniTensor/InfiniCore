#ifndef __MAX_CPU_H__
#define __MAX_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include <algorithm>

ELEMENTWISE_DESCRIPTOR(max, cpu)

namespace op::max::cpu {
typedef struct MaxOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    T operator()(const T &a, const T &b) const {
        return std::max(a, b);
    }
} MaxOp;
} // namespace op::max::cpu

#endif // __MAX_CPU_H__
