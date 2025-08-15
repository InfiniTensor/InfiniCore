#ifndef __SIN_CPU_H__
#define __SIN_CPU_H__

#include <cmath>
#include <type_traits>

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "../../../../utils/custom_types.h"

ELEMENTWISE_DESCRIPTOR(sin, cpu)

namespace op::sin::cpu {
typedef struct SinOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &x) const {
        return std::sin(x);
    }
    
    // 为bf16类型特化，使用double作为中间计算类型以提高精度
    bf16_t operator()(const bf16_t &x) const {
        // 将bf16转换为double进行计算，然后再转回bf16
        double x_double = static_cast<double>(_bf16_to_f32(x));
        double result = std::sin(x_double);
        // 使用utils::cast从double直接转换到bf16，保留更高精度
        return utils::cast<bf16_t>(result);
    }
} SinOp;
} // namespace op::sin::cpu

#endif // __SIN_CPU_H__