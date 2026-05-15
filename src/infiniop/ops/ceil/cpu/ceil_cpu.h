#ifndef __CEIL_CPU_H__
#define __CEIL_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "../../../elementwise/unary.h"

UNARY_ELEMENTWISE_DESCRIPTOR(ceil, cpu, op::elementwise::unary::UnaryMode::Ceil)

#endif // __CEIL_CPU_H__
