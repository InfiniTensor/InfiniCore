#ifndef __HARDSWISH_CPU_H__
#define __HARDSWISH_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

// 1. 定义 Descriptor 类，这里使用宏自动生成
// 对应 namespace op::hardswish::cpu
ELEMENTWISE_DESCRIPTOR(hardswish, cpu)

#include <algorithm> // for std::max, std::min
#include <cmath>

namespace op::hardswish::cpu {

// 2. 定义计算 Functor
typedef struct HardSwishOp {
public:
    // HardSwish 是单输入算子
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &x) const {
        const float x_f = utils::cast<float>(x);
        const float clamped = std::min(std::max(x_f + 3.0f, 0.0f), 6.0f);
        const float result = x_f * clamped * (1.0f / 6.0f);
        return utils::cast<T>(result);
    }
} HardSwishOp;

typedef struct HardSwishContiguousOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &x) const {
        // HardSwish 公式: x * ReLU6(x + 3) / 6
        // 展开: x * min(max(x + 3, 0), 6) / 6

        // 定义常量，确保类型转换正确
        T three = static_cast<T>(3);
        T zero = static_cast<T>(0);
        T six = static_cast<T>(6);
        // 使用乘法代替除法以提高性能 (1/6)
        T scale = static_cast<T>(0.16666667f);

        // 1. 计算 x + 3
        T val = x + three;

        // 2. 计算 ReLU6: clamp(val, 0, 6)
        // 先 max 0，再 min 6
        val = std::max(zero, val); // ReLU
        val = std::min(six, val);  // Cap at 6

        // 3. 最终乘积
        return x * val * scale;
    }
} HardSwishContiguousOp;

} // namespace op::hardswish::cpu

#endif // __HARDSWISH_CPU_H__
