#ifndef __WHERE_CPU_H__
#define __WHERE_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(where, cpu)

namespace op::where::cpu {

struct WhereOp {
    static constexpr size_t num_inputs = 3; // a, b, condition

    // 主要的operator()函数，处理所有数据类型
    template <typename T>
    T operator()(const T &a_val, const T &b_val, const bool &cond) const {
        return cond ? a_val : b_val;
    }

    // 为Metax兼容性添加的模板operator()函数
    template <typename Tout, typename... Tin>
    Tout operator()(const Tin&... args) const {
        static_assert(sizeof...(Tin) == 3, "WhereOp expects exactly 3 arguments");
        // Metax传递的参数顺序是: [a, b, condition]
        const auto& a_val = std::get<0>(std::tie(args...));
        const auto& b_val = std::get<1>(std::tie(args...));
        const bool& cond = std::get<2>(std::tie(args...));
        return cond ? a_val : b_val;
    }

    // 为CPU elementwise BF16特殊处理添加的float版本
    template <typename T>
    T operator()(const float &a_val, const float &b_val, const bool &cond) const {
        return cond ? a_val : b_val;
    }
};

} // namespace op::where::cpu

#endif // __WHERE_CPU_H__
