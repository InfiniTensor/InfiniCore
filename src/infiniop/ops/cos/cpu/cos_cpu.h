#ifndef __COS_CPU_H__
#define __COS_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include <cmath>

ELEMENTWISE_DESCRIPTOR(cos, cpu)

namespace op::cos::cpu {
typedef struct CosOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &input) const {
        return std::cos(input);
    }
} CosOp;
} // namespace op::cos::cpu

#endif // __COS_CPU_H__
