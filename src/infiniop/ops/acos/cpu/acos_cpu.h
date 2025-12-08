#ifndef __ACOS_CPU_H__
#define __ACOS_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

// 使用宏声明 Descriptor 类
ELEMENTWISE_DESCRIPTOR(acos, cpu)

#include <cmath>
#include <type_traits>

namespace op::acos::cpu {

typedef struct AcosOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &x) const {
        // 1. 整数类型：必须先转为浮点计算，再转回整数
        // 注意：这会导致结果截断 (例如 acos(0) = 1.57 -> 1)
        if constexpr (std::is_integral_v<T>) {
            return static_cast<T>(std::acos(static_cast<double>(x)));
        } 
        // 2. 标准浮点类型 (float, double)：直接调用 std::acos
        else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
            return std::acos(x);
        }
        // 3. 半精度类型 (fp16, bf16) 及其他：先转 float 计算
        else {
            return static_cast<T>(std::acos(static_cast<float>(x)));
        }
    }
} AcosOp;

} // namespace op::acos::cpu

#endif // __ACOS_CPU_H__