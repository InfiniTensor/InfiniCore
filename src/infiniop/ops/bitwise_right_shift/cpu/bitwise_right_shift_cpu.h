#ifndef __BITWISE_RIGHT_SHIFT_CPU_H__
#define __BITWISE_RIGHT_SHIFT_CPU_H__

#include "../../../elementwise/binary.h"
#include "../../../elementwise/cpu/elementwise_cpu.h"

BINARY_ELEMENTWISE_DESCRIPTOR(bitwise_right_shift, cpu, op::elementwise::binary::BinaryMode::BitwiseRightShift)

#endif // __BITWISE_RIGHT_SHIFT_CPU_H__
