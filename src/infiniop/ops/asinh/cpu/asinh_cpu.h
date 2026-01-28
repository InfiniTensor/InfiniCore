#ifndef __ASINH_CPU_H__
#define __ASINH_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "../../../elementwise/unary.h"

UNARY_ELEMENTWISE_DESCRIPTOR(asinh, cpu, op::elementwise::unary::UnaryMode::Asinh)

#endif // __ASINH_CPU_H__
