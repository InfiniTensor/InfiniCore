#ifndef __ATANH_CPU_H__
#define __ATANH_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "../../../elementwise/unary.h"

UNARY_ELEMENTWISE_DESCRIPTOR(atanh, cpu, op::elementwise::unary::UnaryMode::Atanh)

#endif // __ATANH_CPU_H__
