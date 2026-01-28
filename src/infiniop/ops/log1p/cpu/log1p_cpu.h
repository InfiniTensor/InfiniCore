#ifndef __LOG1P_CPU_H__
#define __LOG1P_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "../../../elementwise/unary.h"

UNARY_ELEMENTWISE_DESCRIPTOR(log1p, cpu, op::elementwise::unary::UnaryMode::Log1p)

#endif // __LOG1P_CPU_H__
