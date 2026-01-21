#ifndef __LOG2_CPU_H__
#define __LOG2_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "../../../elementwise/unary.h"

UNARY_ELEMENTWISE_DESCRIPTOR(log2, cpu, op::elementwise::unary::UnaryMode::Log2)

#endif // __LOG2_CPU_H__
