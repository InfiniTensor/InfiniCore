#ifndef __NE_CPU_H__
#define __NE_CPU_H__

#include "../../../elementwise/binary.h"
#include "../../../elementwise/cpu/elementwise_cpu.h"

BINARY_ELEMENTWISE_DESCRIPTOR(ne, cpu, op::elementwise::binary::BinaryMode::NotEqual)

#endif // __NE_CPU_H__
