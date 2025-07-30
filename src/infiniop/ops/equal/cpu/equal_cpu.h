#ifndef __EQUAL_CPU_H__
#define __EQUAL_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(equal, cpu)

namespace op::equal::cpu {
typedef struct EqualOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename Tout, typename Ta, typename Tb>
    Tout operator()(const Ta &a, const Tb &b) const {
        if constexpr (!std::is_same_v<Ta, Tb>) {
            printf("Ta and Tb must be the same type!\n");
            std::abort();
        }
        if constexpr (std::is_same_v<Ta, bf16_t> || std::is_same_v<Ta, fp16_t>) {
            float f_a = utils::cast<float, Ta>(a);
            float f_b = utils::cast<float, Ta>(b);
            return f_a == f_b;
        } else {
            return a == b;
        }
    }
} EqualOp;
} // namespace op::equal::cpu

#endif // __EQUAL_CPU_H__
