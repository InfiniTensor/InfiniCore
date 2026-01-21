#ifndef __COPYSIGN_CPU_H__
#define __COPYSIGN_CPU_H__

#include "../../../elementwise/binary.h"
#include "../../../elementwise/cpu/elementwise_cpu.h"

BINARY_ELEMENTWISE_DESCRIPTOR(copysign, cpu, op::elementwise::binary::BinaryMode::CopySign)

#endif // __COPYSIGN_CPU_H__
