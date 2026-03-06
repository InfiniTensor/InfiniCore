#ifndef __EQUAL_CPU_H__
#define __EQUAL_CPU_H__

#include <type_traits>

#include "../../../elementwise/cpu/elementwise_cpu.h"


ELEMENTWISE_DESCRIPTOR(equal, cpu)

namespace op::equal::cpu {


typedef struct EqualOp {
public:
    static constexpr size_t num_inputs = 2;

    template <typename T>
    
    
    T operator()(const T &a, const T &b) const {
        return static_cast<T>(a == b);
    }

    template <typename Tout, typename Tin0, typename Tin1>
    Tout operator()(const Tin0 &a, const Tin1 &b) const {
        static_assert(std::is_same_v<Tin0, Tin1>, "EqualOp expects identical input dtypes");
        return static_cast<Tout>(a == b);
    }
} EqualOp;

} 

#endif 
