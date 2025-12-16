#ifndef __LOGSIGMOID_CPU_H__
#define __LOGSIGMOID_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(logsigmoid, cpu)

namespace op::logsigmoid::cpu {
typedef struct LogSigmoidOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    T operator()(const T &x) const {
        // logsigmoid(x) = log(sigmoid(x)) = -log(1 + exp(-x))
        return -std::log(T(1) + std::exp(-x));
    }
} LogSigmoidOp;
} // namespace op::logsigmoid::cpu

#endif // __LOGSIGMOID_CPU_H__
