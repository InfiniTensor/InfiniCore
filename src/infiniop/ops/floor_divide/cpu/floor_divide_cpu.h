#ifndef __FLOOR_DIVIDE_CPU_H__
#define __FLOOR_DIVIDE_CPU_H__

#include "../../../elementwise/binary.h"
#include "../../../elementwise/cpu/elementwise_cpu.h"

BINARY_ELEMENTWISE_DESCRIPTOR(floor_divide, cpu, op::elementwise::binary::BinaryMode::FloorDivide)

#endif // __FLOOR_DIVIDE_CPU_H__
