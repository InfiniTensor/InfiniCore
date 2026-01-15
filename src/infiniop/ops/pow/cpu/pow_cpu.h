#ifndef __POW_CPU_H__
#define __POW_CPU_H__

#include "../../../elementwise/binary.h"
#include "../../../elementwise/cpu/elementwise_cpu.h"

BINARY_ELEMENTWISE_DESCRIPTOR(pow, cpu, op::elementwise::binary::BinaryMode::Pow)

#endif // __POW_CPU_H__
