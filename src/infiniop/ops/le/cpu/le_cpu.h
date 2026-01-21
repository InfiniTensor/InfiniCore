#ifndef __LE_CPU_H__
#define __LE_CPU_H__

#include "../../../elementwise/binary.h"
#include "../../../elementwise/cpu/elementwise_cpu.h"

BINARY_ELEMENTWISE_DESCRIPTOR(le, cpu, op::elementwise::binary::BinaryMode::LessOrEqual)

#endif // __LE_CPU_H__
