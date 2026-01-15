#ifndef __MAX_CPU_H__
#define __MAX_CPU_H__

#include "../../../elementwise/binary.h"
#include "../../../elementwise/cpu/elementwise_cpu.h"

BINARY_ELEMENTWISE_DESCRIPTOR(max, cpu, op::elementwise::binary::BinaryMode::Max)

#endif // __MAX_CPU_H__
