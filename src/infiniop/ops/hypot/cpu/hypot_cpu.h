#ifndef __HYPOT_CPU_H__
#define __HYPOT_CPU_H__

#include "../../../elementwise/binary.h"
#include "../../../elementwise/cpu/elementwise_cpu.h"

BINARY_ELEMENTWISE_DESCRIPTOR(hypot, cpu, op::elementwise::binary::BinaryMode::Hypot)

#endif // __HYPOT_CPU_H__
