#ifndef __LOGICAL_OR_CPU_H__
#define __LOGICAL_OR_CPU_H__

#include "../../../elementwise/binary.h"
#include "../../../elementwise/cpu/elementwise_cpu.h"

BINARY_ELEMENTWISE_DESCRIPTOR(logical_or, cpu, op::elementwise::binary::BinaryMode::LogicalOr)

#endif // __LOGICAL_OR_CPU_H__
