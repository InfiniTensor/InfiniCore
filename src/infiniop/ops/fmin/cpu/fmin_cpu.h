#ifndef __FMIN_CPU_H__
#define __FMIN_CPU_H__

#include "../../../elementwise/binary.h"
#include "../../../elementwise/cpu/elementwise_cpu.h"

BINARY_ELEMENTWISE_DESCRIPTOR(fmin, cpu, op::elementwise::binary::BinaryMode::Fmin)

#endif // __FMIN_CPU_H__
