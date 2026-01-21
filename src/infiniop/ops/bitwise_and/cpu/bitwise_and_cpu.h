#ifndef __BITWISE_AND_CPU_H__
#define __BITWISE_AND_CPU_H__

#include "../../../elementwise/binary.h"
#include "../../../elementwise/cpu/elementwise_cpu.h"

BINARY_ELEMENTWISE_DESCRIPTOR(bitwise_and, cpu, op::elementwise::binary::BinaryMode::BitwiseAnd)

#endif // __BITWISE_AND_CPU_H__
