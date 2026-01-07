#ifndef __ASIN_CPU_H__
#define __ASIN_CPU_H__

#include <cmath>

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(asin, cpu)

namespace op::asin::cpu {
typedef struct AsinOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &x) const {
        return std::asin(x);
    }
} AsinOp;
} // namespace op::asin::cpu

#endif // __ASIN_CPU_H__
