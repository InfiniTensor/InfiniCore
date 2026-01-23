#ifndef __SIGN_CPU_H__
#define __SIGN_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "../../../elementwise/unary.h"

UNARY_ELEMENTWISE_DESCRIPTOR(sign, cpu, op::elementwise::unary::UnaryMode::Sign)

#endif // __SIGN_CPU_H__
