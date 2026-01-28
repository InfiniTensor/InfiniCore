#ifndef __ISINF_CPU_H__
#define __ISINF_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "../../../elementwise/unary.h"

UNARY_ELEMENTWISE_DESCRIPTOR(isinf, cpu, op::elementwise::unary::UnaryMode::IsInf)

#endif // __ISINF_CPU_H__
