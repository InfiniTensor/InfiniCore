#ifndef __MOD_CPU_H__
#define __MOD_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(mod, cpu)

namespace op::mod::cpu {
typedef struct ModOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    T operator()(const T &a, const T &b) const {
        if constexpr (std::is_floating_point_v<T>) {
            return std::fmod(a, b);
        } else {
            return a % b;
        }
    }
} ModOp;
} // namespace op::mod::cpu

#endif // __MOD_CPU_H__
