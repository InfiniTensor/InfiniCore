#ifndef __HARDTANH_CPU_H__
#define __HARDTANH_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include <algorithm> // 用于 std::max 和 std::min

// 注册算子描述符
ELEMENTWISE_DESCRIPTOR(hardtanh, cpu)

namespace op::hardtanh::cpu {

typedef struct HardTanhOp {
public:
    static constexpr size_t num_inputs = 1;

    // 存储算子状态（截断范围）
    float min_val;
    float max_val;

    // 构造函数，用于从 Descriptor 中初始化参数
    HardTanhOp(float min_v = -1.0f, float max_v = 1.0f) 
        : min_val(min_v), max_val(max_v) {}

    template <typename T>
    T operator()(const T &x) const {
        // 使用标准库的 clamp 逻辑
        T val = x < static_cast<T>(min_val) ? static_cast<T>(min_val) : x;
        return val > static_cast<T>(max_val) ? static_cast<T>(max_val) : val;
    }
} HardTanhOp;

} // namespace op::hardtanh::cpu

#endif // __HARDTANH_CPU_H__