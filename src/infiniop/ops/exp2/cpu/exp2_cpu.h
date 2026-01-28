#ifndef __EXP2_CPU_H__
#define __EXP2_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "../../../elementwise/unary.h"

UNARY_ELEMENTWISE_DESCRIPTOR(exp2, cpu, op::elementwise::unary::UnaryMode::Exp2)

#endif // __EXP2_CPU_H__
