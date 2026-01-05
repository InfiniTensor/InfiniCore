#ifndef __MIN_CPU_H__
#define __MIN_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include <algorithm>

ELEMENTWISE_DESCRIPTOR(min, cpu)

namespace op::min::cpu {
typedef struct MinOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    T operator()(const T &a, const T &b) const {
        return std::min(a, b);
    }
} MinOp;
} // namespace op::min::cpu

#endif // __MIN_CPU_H__
