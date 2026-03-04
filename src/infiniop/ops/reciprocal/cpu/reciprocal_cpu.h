#ifndef __RECIPROCAL_CPU_H__
#define __RECIPROCAL_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "../../../elementwise/unary.h"

UNARY_ELEMENTWISE_DESCRIPTOR(reciprocal, cpu, op::elementwise::unary::UnaryMode::Reciprocal)

#endif // __RECIPROCAL_CPU_H__
