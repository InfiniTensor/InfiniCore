#ifndef __COS_CPU_H__
#define __COS_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "../../../elementwise/unary.h"

UNARY_ELEMENTWISE_DESCRIPTOR(cos, cpu, op::elementwise::unary::UnaryMode::Cos)

#endif // __COS_CPU_H__
