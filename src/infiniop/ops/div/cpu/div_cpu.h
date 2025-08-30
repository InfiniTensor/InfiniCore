#ifndef __DIV_CPU_H__
#define __DIV_CPU_H__

#include <cmath>
#include <type_traits>
#include <limits>

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "../../../../utils/custom_types.h"

ELEMENTWISE_DESCRIPTOR(div, cpu)

namespace op::div::cpu {
typedef struct DivOp {
public:
    static constexpr size_t num_inputs = 2;

    template <typename T>
    T operator()(const T &a, const T &b) const {
        // 添加除零保护
        if (b == static_cast<T>(0)) {
            if (a > static_cast<T>(0)) {
                return std::numeric_limits<T>::infinity();
            } else if (a < static_cast<T>(0)) {
                return -std::numeric_limits<T>::infinity();
            } else {
                return std::numeric_limits<T>::quiet_NaN();
            }
        }
        return a / b;
    }
    
    // 为bf16类型特化，使用double作为中间计算类型以提高精度
    bf16_t operator()(const bf16_t &a, const bf16_t &b) const {
        // 将bf16转换为double进行计算，然后再转回bf16
        double a_double = static_cast<double>(_bf16_to_f32(a));
        double b_double = static_cast<double>(_bf16_to_f32(b));
        
        // 添加除零保护
        if (b_double == 0.0) {
            if (a_double > 0.0) {
                return utils::cast<bf16_t>(std::numeric_limits<double>::infinity());
            } else if (a_double < 0.0) {
                return utils::cast<bf16_t>(-std::numeric_limits<double>::infinity());
            } else {
                return utils::cast<bf16_t>(std::numeric_limits<double>::quiet_NaN());
            }
        }
        
        double result = a_double / b_double;
        // 使用utils::cast从double直接转换到bf16，保留更高精度
        return utils::cast<bf16_t>(result);
    }
} DivOp;
} // namespace op::div::cpu

#endif // __DIV_CPU_H__