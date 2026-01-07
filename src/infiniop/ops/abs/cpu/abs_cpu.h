#ifndef __ABS_CPU_H__
#define __ABS_CPU_H__

#include <cmath>

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(abs, cpu)

namespace op::abs::cpu {
typedef struct AbsOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &x) const {
        if constexpr (std::is_floating_point_v<T>) {
            return std::fabs(x);
        } else {
            return std::abs(x);
        }
    }
} AbsOp;
} // namespace op::abs::cpu

#endif // __ABS_CPU_H__
