#ifndef __SOFTPLUS_CPU_H__
#define __SOFTPLUS_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(softplus, cpu)

namespace op::softplus::cpu {
typedef struct SoftplusOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    T operator()(const T &x) const {
        if (x > T(20)) {
            return x;
        } else {
            return std::log(T(1) + std::exp(x));
        }
    }
} SoftplusOp;
} // namespace op::softplus::cpu

#endif // __SOFTPLUS_CPU_H__
