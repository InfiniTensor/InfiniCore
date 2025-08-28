#ifndef __CAST_CPU_H__
#define __CAST_CPU_H__

#include "../cast.h"
#include "../../../elementwise/cpu/elementwise_cpu.h"

#include "../../../../utils/custom_types.h"

DESCRIPTOR(cpu)

namespace op::cast::cpu {


typedef struct CastOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename TypeTo, typename TypeFrom>
    TypeTo operator()(const TypeFrom &val) const {
        return utils::cast<TypeTo, TypeFrom>(val);
    };
} CastOp;
}

#endif // __CAST_CPU_H__
