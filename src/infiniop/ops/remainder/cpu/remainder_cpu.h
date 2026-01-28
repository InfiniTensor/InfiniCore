#ifndef __REMAINDER_CPU_H__
#define __REMAINDER_CPU_H__

#include "../../../elementwise/binary.h"
#include "../../../elementwise/cpu/elementwise_cpu.h"

BINARY_ELEMENTWISE_DESCRIPTOR(remainder, cpu, op::elementwise::binary::BinaryMode::Remainder)

#endif // __REMAINDER_CPU_H__
