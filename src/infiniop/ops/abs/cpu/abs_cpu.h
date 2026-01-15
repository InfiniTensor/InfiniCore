#ifndef __ABS_CPU_H__
#define __ABS_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "../../../elementwise/unary.h"

UNARY_ELEMENTWISE_DESCRIPTOR(abs, cpu, op::elementwise::unary::UnaryMode::Abs)

#endif // __ABS_CPU_H__
