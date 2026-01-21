#ifndef __BITWISE_OR_CPU_H__
#define __BITWISE_OR_CPU_H__

#include "../../../elementwise/binary.h"
#include "../../../elementwise/cpu/elementwise_cpu.h"

BINARY_ELEMENTWISE_DESCRIPTOR(bitwise_or, cpu, op::elementwise::binary::BinaryMode::BitwiseOr)

#endif // __BITWISE_OR_CPU_H__
