#ifndef __RSQRT_CPU_H__
#define __RSQRT_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "../../../elementwise/unary.h"

UNARY_ELEMENTWISE_DESCRIPTOR(rsqrt, cpu, op::elementwise::unary::UnaryMode::Rsqrt)

#endif // __RSQRT_CPU_H__
