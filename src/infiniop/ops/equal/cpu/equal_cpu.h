#ifndef __EQUAL_CPU_H__
#define __EQUAL_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(equal_op, cpu)

namespace op::equal_op::cpu {
typedef struct EqualOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T, typename Ta, typename Tb>
    T operator()(const Ta &a, const Tb &b) const {
        // Logical AND: non-zero values are treated as true
        if constexpr (std::is_same_v<Ta, bool> && std::is_same_v<Tb, bool>) {
            return a == b;
        } else if constexpr (std::is_same_v<Ta, bf16_t> || std::is_same_v<Tb, fp16_t>) {
            // For bf16 and fp16, we can use the cast to float for comparison
            return static_cast<T>(utils::cast<float>(a) == utils::cast<float>(b));
        } else {
            return static_cast<T>(a == b);
        }
    }
} EqualOp;

} // namespace op::equal_op::cpu

#endif // __EQUAL_CPU_H__