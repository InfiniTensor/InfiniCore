#ifndef __EXP_CPU_H__
#define __EXP_CPU_H__

#include <cmath>
#include <type_traits>

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "../../../../utils/custom_types.h"

ELEMENTWISE_DESCRIPTOR(exp, cpu)

namespace op::exp::cpu {
typedef struct ExpOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &x) const {
        return std::exp(x);
    }
    
    // 为bf16类型特化，使用double作为中间计算类型以提高精度
    bf16_t operator()(const bf16_t &x) const {
        // 将bf16转换为double进行计算，然后再转回bf16
        double x_double = static_cast<double>(_bf16_to_f32(x));
        double result = std::exp(x_double);
        // 使用utils::cast从double直接转换到bf16，保留更高精度
        return utils::cast<bf16_t>(result);
    }
} ExpOp;
} // namespace op::exp::cpu

#endif // __EXP_CPU_H__