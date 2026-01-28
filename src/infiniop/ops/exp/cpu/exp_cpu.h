#ifndef __EXP_CPU_H__
#define __EXP_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "../../../elementwise/unary.h"

UNARY_ELEMENTWISE_DESCRIPTOR(exp, cpu, op::elementwise::unary::UnaryMode::Exp)

#endif // __EXP_CPU_H__
