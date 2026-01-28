#ifndef __LT_CPU_H__
#define __LT_CPU_H__

#include "../../../elementwise/binary.h"
#include "../../../elementwise/cpu/elementwise_cpu.h"

BINARY_ELEMENTWISE_DESCRIPTOR(lt, cpu, op::elementwise::binary::BinaryMode::Less)

#endif // __LT_CPU_H__
