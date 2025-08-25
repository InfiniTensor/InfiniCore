#ifndef __HARDSWISH_CPU_H__
#define __HARDSWISH_CPU_H__

#include <cmath>
#include <type_traits>
#include <algorithm>

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "../../../../utils/custom_types.h"

ELEMENTWISE_DESCRIPTOR(hardswish, cpu)

namespace op::hardswish::cpu {
typedef struct HardSwishOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &x) const {
        // HardSwish: x * ReLU6(x + 3) / 6
        // ReLU6(x) = min(max(x, 0), 6)
        T relu6_input = x + static_cast<T>(3.0);
        T relu6_output = std::min(std::max(relu6_input, static_cast<T>(0.0)), static_cast<T>(6.0));
        return x * relu6_output / static_cast<T>(6.0);
    }
    
    // 为bf16类型特化，使用double作为中间计算类型以提高精度
    bf16_t operator()(const bf16_t &x) const {
        // 将bf16转换为double进行计算，然后再转回bf16
        double x_double = static_cast<double>(_bf16_to_f32(x));
        // HardSwish: x * ReLU6(x + 3) / 6
        double relu6_input = x_double + 3.0;
        double relu6_output = std::min(std::max(relu6_input, 0.0), 6.0);
        double result = x_double * relu6_output / 6.0;
        // 使用utils::cast从double直接转换到bf16，保留更高精度
        return utils::cast<bf16_t>(result);
    }
} HardSwishOp;
} // namespace op::hardswish::cpu

#endif // __HARDSWISH_CPU_H__