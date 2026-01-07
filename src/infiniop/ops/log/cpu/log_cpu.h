#ifndef __LOG_CPU_H__
#define __LOG_CPU_H__

#include <cmath>

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(log, cpu)

namespace op::log::cpu {
typedef struct LogOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &x) const {
        return std::log(x);
    }
} LogOp;
} // namespace op::log::cpu

#endif // __LOG_CPU_H__
