#ifndef __SINH_CPU_H__
#define __SINH_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "../../../elementwise/unary.h"

UNARY_ELEMENTWISE_DESCRIPTOR(sinh, cpu, op::elementwise::unary::UnaryMode::Sinh)

#endif // __SINH_CPU_H__
