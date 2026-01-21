#ifndef __ISFINITE_CPU_H__
#define __ISFINITE_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "../../../elementwise/unary.h"

UNARY_ELEMENTWISE_DESCRIPTOR(isfinite, cpu, op::elementwise::unary::UnaryMode::IsFinite)

#endif // __ISFINITE_CPU_H__
