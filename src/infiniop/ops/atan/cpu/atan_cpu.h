#ifndef __ATAN_CPU_H__
#define __ATAN_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "../../../elementwise/unary.h"

UNARY_ELEMENTWISE_DESCRIPTOR(atan, cpu, op::elementwise::unary::UnaryMode::Atan)

#endif // __ATAN_CPU_H__
