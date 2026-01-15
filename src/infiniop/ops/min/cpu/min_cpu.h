#ifndef __MIN_CPU_H__
#define __MIN_CPU_H__

#include "../../../elementwise/binary.h"
#include "../../../elementwise/cpu/elementwise_cpu.h"

BINARY_ELEMENTWISE_DESCRIPTOR(min, cpu, op::elementwise::binary::BinaryMode::Min)

#endif // __MIN_CPU_H__
