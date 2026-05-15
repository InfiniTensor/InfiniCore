#ifndef __DIV_CPU_H__
#define __DIV_CPU_H__

#include "../../../elementwise/binary.h"
#include "../../../elementwise/cpu/elementwise_cpu.h"

BINARY_ELEMENTWISE_DESCRIPTOR(div, cpu, op::elementwise::binary::BinaryMode::Divide)

#endif // __DIV_CPU_H__
