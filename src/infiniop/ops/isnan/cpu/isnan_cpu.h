#ifndef __ISNAN_CPU_H__
#define __ISNAN_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "../../../elementwise/unary.h"

UNARY_ELEMENTWISE_DESCRIPTOR(isnan, cpu, op::elementwise::unary::UnaryMode::IsNan)

#endif // __ISNAN_CPU_H__
