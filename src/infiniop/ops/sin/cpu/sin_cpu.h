#ifndef __SIN_CPU_H__
#define __SIN_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "../../../elementwise/unary.h"

UNARY_ELEMENTWISE_DESCRIPTOR(sin, cpu, op::elementwise::unary::UnaryMode::Sin)

#endif // __SIN_CPU_H__
