#ifndef __FMAX_CPU_H__
#define __FMAX_CPU_H__

#include "../../../elementwise/binary.h"
#include "../../../elementwise/cpu/elementwise_cpu.h"

BINARY_ELEMENTWISE_DESCRIPTOR(fmax, cpu, op::elementwise::binary::BinaryMode::Fmax)

#endif // __FMAX_CPU_H__
