#ifndef __LOGICAL_AND_CPU_H__
#define __LOGICAL_AND_CPU_H__

#include "../../../elementwise/binary.h"
#include "../../../elementwise/cpu/elementwise_cpu.h"

BINARY_ELEMENTWISE_DESCRIPTOR(logical_and, cpu, op::elementwise::binary::BinaryMode::LogicalAnd)

#endif // __LOGICAL_AND_CPU_H__
