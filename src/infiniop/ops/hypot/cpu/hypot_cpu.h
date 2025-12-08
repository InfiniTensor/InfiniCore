#ifndef __HYPOT_CPU_H__
#define __HYPOT_CPU_H__

// 引入基础宏定义
#include "../../../elementwise/cpu/elementwise_cpu.h"

// 使用宏声明 Descriptor 类 (op::hypot::cpu)
ELEMENTWISE_DESCRIPTOR(hypot, cpu)

#include <cmath>
#include <type_traits>

namespace op::hypot::cpu {

typedef struct HypotOp {
public:
    // Hypot 是二元算子，计算 sqrt(x^2 + y^2)
    static constexpr size_t num_inputs = 2;

    template <typename T>
    T operator()(const T &x, const T &y) const {
        // 1. 标准浮点类型 (float, double)：直接调用 std::hypot，保证精度
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
            return std::hypot(x, y);
        }
        // 2. 其他类型：
        //    - 半精度 (fp16, bf16)：先转 float 计算
        //    - 整数类型：通常 hypot 结果为浮点数，如果 T 是整数，这里会发生截断 (例如 1.414 -> 1)
        else {
            return static_cast<T>(std::hypot(static_cast<float>(x), static_cast<float>(y)));
        }
    }
} HypotOp;

} // namespace op::hypot::cpu

#endif // __HYPOT_CPU_H__