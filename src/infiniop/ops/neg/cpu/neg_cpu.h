#ifndef __NEG_CPU_H__
#define __NEG_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "../../../elementwise/unary.h"

UNARY_ELEMENTWISE_DESCRIPTOR(neg, cpu, op::elementwise::unary::UnaryMode::Neg)

#endif // __NEG_CPU_H__
