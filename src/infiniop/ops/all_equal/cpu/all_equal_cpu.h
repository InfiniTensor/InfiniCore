#ifndef __ALL_EQUAL_CPU_H__
#define __ALL_EQUAL_CPU_H__

#include "../all_equal.h"

DESCRIPTOR(cpu)

namespace op::all_equal::cpu {
typedef struct AllEqualOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename Tout, typename Tin>
    Tout operator()(const Tin &a, const Tin &b) const {
        return a == b;
    }
} AllEqualOp;
} // namespace op::all_equal::cpu

#endif // __ALL_EQUAL_CPU_H__
