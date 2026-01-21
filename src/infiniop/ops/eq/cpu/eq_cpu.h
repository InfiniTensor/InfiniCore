#ifndef __EQ_CPU_H__
#define __EQ_CPU_H__

#include "../../../elementwise/binary.h"
#include "../../../elementwise/cpu/elementwise_cpu.h"

BINARY_ELEMENTWISE_DESCRIPTOR(eq, cpu, op::elementwise::binary::BinaryMode::Equal)

#endif // __EQ_CPU_H__
