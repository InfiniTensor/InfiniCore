#ifndef __MOD_CPU_H__
#define __MOD_CPU_H__

#include "../../../elementwise/binary.h"
#include "../../../elementwise/cpu/elementwise_cpu.h"

BINARY_ELEMENTWISE_DESCRIPTOR(mod, cpu, op::elementwise::binary::BinaryMode::Mod)

#endif // __MOD_CPU_H__
