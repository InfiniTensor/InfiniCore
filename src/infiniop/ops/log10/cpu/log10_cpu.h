#ifndef __LOG10_CPU_H__
#define __LOG10_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "../../../elementwise/unary.h"

UNARY_ELEMENTWISE_DESCRIPTOR(log10, cpu, op::elementwise::unary::UnaryMode::Log10)

#endif // __LOG10_CPU_H__
