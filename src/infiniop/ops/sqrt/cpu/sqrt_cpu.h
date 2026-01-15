#ifndef __SQRT_CPU_H__
#define __SQRT_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "../../../elementwise/unary.h"

UNARY_ELEMENTWISE_DESCRIPTOR(sqrt, cpu, op::elementwise::unary::UnaryMode::Sqrt)

#endif // __SQRT_CPU_H__
