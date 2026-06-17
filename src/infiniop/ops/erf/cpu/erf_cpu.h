#ifndef __ERF_CPU_H__
#define __ERF_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "../../../elementwise/unary.h"

UNARY_ELEMENTWISE_DESCRIPTOR(erf, cpu, op::elementwise::unary::UnaryMode::Erf)

#endif // __ERF_CPU_H__
