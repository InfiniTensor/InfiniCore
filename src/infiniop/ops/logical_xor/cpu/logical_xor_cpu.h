#ifndef __LOGICAL_XOR_CPU_H__
#define __LOGICAL_XOR_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(logical_xor, cpu)

namespace op::logical_xor::cpu {
typedef struct LogicalXorOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    bool operator()(const T &a, const T &b) const {
        return static_cast<bool>(a) != static_cast<bool>(b);
    }
    // Support heterogeneous input types for elementwise framework
    template <typename Tout, typename Ta, typename Tb>
    Tout operator()(const Ta &a, const Tb &b) const {
        return static_cast<Tout>(static_cast<bool>(a) != static_cast<bool>(b));
    }
} LogicalXorOp;
} // namespace op::logical_xor::cpu

#endif // __LOGICAL_XOR_CPU_H__
