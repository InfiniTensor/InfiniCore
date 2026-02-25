#ifndef __TAN_CPU_H__
#define __TAN_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "../../../elementwise/unary.h"

UNARY_ELEMENTWISE_DESCRIPTOR(tan, cpu, op::elementwise::unary::UnaryMode::Tan)

#endif // __TAN_CPU_H__
