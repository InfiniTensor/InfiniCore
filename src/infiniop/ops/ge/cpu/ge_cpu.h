#ifndef __GE_CPU_H__
#define __GE_CPU_H__

#include "../../../elementwise/binary.h"
#include "../../../elementwise/cpu/elementwise_cpu.h"

BINARY_ELEMENTWISE_DESCRIPTOR(ge, cpu, op::elementwise::binary::BinaryMode::GreaterOrEqual)

#endif // __GE_CPU_H__
