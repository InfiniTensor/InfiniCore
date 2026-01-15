#ifndef __COSH_CPU_H__
#define __COSH_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "../../../elementwise/unary.h"

UNARY_ELEMENTWISE_DESCRIPTOR(cosh, cpu, op::elementwise::unary::UnaryMode::Cosh)

#endif // __COSH_CPU_H__
