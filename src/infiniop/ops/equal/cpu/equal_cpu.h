#ifndef __EQUAL_CPU_H__
#define __EQUAL_CPU_H__

#include <type_traits>

#include "../../../elementwise/cpu/elementwise_cpu.h"

// 自动生成 Descriptor 类声明
ELEMENTWISE_DESCRIPTOR(equal, cpu)

namespace op::equal::cpu {

// 定义核心计算 Functor
typedef struct EqualOp {
public:
    static constexpr size_t num_inputs = 2;

    template <typename T>
    // 这里返回 T 类型是为了适配通用 Elementwise 模板的签名
    // 实际计算 a == b 会返回 bool，然后 static_cast 转回 T (如 1.0 或 0.0)
    T operator()(const T &a, const T &b) const {
        return static_cast<T>(a == b);
    }

    template <typename Tout, typename Tin0, typename Tin1>
    Tout operator()(const Tin0 &a, const Tin1 &b) const {
        static_assert(std::is_same_v<Tin0, Tin1>, "EqualOp expects identical input dtypes");
        return static_cast<Tout>(a == b);
    }
} EqualOp;

} // namespace op::equal::cpu

#endif // __EQUAL_CPU_H__
