#ifndef __ACOSH_CPU_H__
#define __ACOSH_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "../../../elementwise/unary.h"

UNARY_ELEMENTWISE_DESCRIPTOR(acosh, cpu, op::elementwise::unary::UnaryMode::Acosh)

#endif // __ACOSH_CPU_H__
