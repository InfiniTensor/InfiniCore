#ifndef __BITWISE_XOR_CPU_H__
#define __BITWISE_XOR_CPU_H__

#include "../../../elementwise/binary.h"
#include "../../../elementwise/cpu/elementwise_cpu.h"

BINARY_ELEMENTWISE_DESCRIPTOR(bitwise_xor, cpu, op::elementwise::binary::BinaryMode::BitwiseXor)

#endif // __BITWISE_XOR_CPU_H__
