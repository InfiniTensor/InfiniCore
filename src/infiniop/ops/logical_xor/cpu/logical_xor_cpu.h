#ifndef __LOGICAL_XOR_CPU_H__
#define __LOGICAL_XOR_CPU_H__

#include "../../../elementwise/binary.h"
#include "../../../elementwise/cpu/elementwise_cpu.h"

BINARY_ELEMENTWISE_DESCRIPTOR(logical_xor, cpu, op::elementwise::binary::BinaryMode::LogicalXor)

#endif // __LOGICAL_XOR_CPU_H__
