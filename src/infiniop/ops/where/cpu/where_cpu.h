#ifndef __WHERE_CPU_H__
#define __WHERE_CPU_H__

#include <type_traits>

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "../../../../utils/custom_types.h"

ELEMENTWISE_DESCRIPTOR(where, cpu)

namespace op::where::cpu {
typedef struct WhereOp {
public:
    static constexpr size_t num_inputs = 3;

    // 异构输入类型的operator，用于处理condition(bool)和a,b(float等)不同类型的情况
    // 注意：根据elementwise框架，参数顺序应该与inputs向量顺序一致：inputs[0]=condition, inputs[1]=a, inputs[2]=b
    template <typename Tout, typename Tcond, typename Ta, typename Tb>
    Tout operator()(const Tcond &condition, const Ta &a, const Tb &b) const {
        bool cond_bool;
        if constexpr (std::is_same_v<Tcond, bool>) {
            cond_bool = condition;
        } else {
            // 假设是int8类型表示bool
            cond_bool = (condition != 0);
        }
        
        return cond_bool ? static_cast<Tout>(a) : static_cast<Tout>(b);
    }
} WhereOp;
} // namespace op::where::cpu

#endif // __WHERE_CPU_H__