#ifndef __SIGN_CPU_H__
#define __SIGN_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(sign, cpu)

namespace op::sign::cpu {
typedef struct SignOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &x) const {
        return x > T(0) ? T(1) : (x == T(0) ? T(0) : T(-1));
    }
} SignOp;
} // namespace op::sign::cpu

#endif // __SIGN_CPU_H__
