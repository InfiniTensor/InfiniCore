#ifndef __BITWISE_LEFT_SHIFT_CPU_H__
#define __BITWISE_LEFT_SHIFT_CPU_H__

#include "../../../elementwise/binary.h"
#include "../../../elementwise/cpu/elementwise_cpu.h"

BINARY_ELEMENTWISE_DESCRIPTOR(bitwise_left_shift, cpu, op::elementwise::binary::BinaryMode::BitwiseLeftShift)

#endif // __BITWISE_LEFT_SHIFT_CPU_H__
