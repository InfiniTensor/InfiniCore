#ifndef __GT_CPU_H__
#define __GT_CPU_H__

#include "../../../elementwise/binary.h"
#include "../../../elementwise/cpu/elementwise_cpu.h"

BINARY_ELEMENTWISE_DESCRIPTOR(gt, cpu, op::elementwise::binary::BinaryMode::Greater)

#endif // __GT_CPU_H__
