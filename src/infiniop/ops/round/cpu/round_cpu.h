#ifndef __ROUND_CPU_H__
#define __ROUND_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "../../../elementwise/unary.h"

UNARY_ELEMENTWISE_DESCRIPTOR(round, cpu, op::elementwise::unary::UnaryMode::Round)

#endif // __ROUND_CPU_H__
