#ifndef __ATAN2_CPU_H__
#define __ATAN2_CPU_H__

#include "../../../elementwise/binary.h"
#include "../../../elementwise/cpu/elementwise_cpu.h"

BINARY_ELEMENTWISE_DESCRIPTOR(atan2, cpu, op::elementwise::binary::BinaryMode::Atan2)

#endif // __ATAN2_CPU_H__
