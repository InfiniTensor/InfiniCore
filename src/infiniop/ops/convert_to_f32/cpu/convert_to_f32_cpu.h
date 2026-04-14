#ifndef __CONVERT_TO_F32_CPU_H__
#define __CONVERT_TO_F32_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(convert_to_f32, cpu)

namespace op::convert_to_f32::cpu {
typedef struct ConvertToF32Op {
public:
    static constexpr size_t num_inputs = 1;
    template <typename Tout, typename Tin>
    Tout operator()(const Tin &x) const {
        return utils::cast<Tout>(x);
    }
} ConvertToF32Op;
} // namespace op::convert_to_f32::cpu

#endif // __CONVERT_TO_F32_CPU_H__
