#ifndef __ACOS_CPU_H__
#define __ACOS_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "../../../elementwise/unary.h"

UNARY_ELEMENTWISE_DESCRIPTOR(acos, cpu, op::elementwise::unary::UnaryMode::Acos)

#endif // __ACOS_CPU_H__
