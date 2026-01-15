#ifndef __LOG_CPU_H__
#define __LOG_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "../../../elementwise/unary.h"

UNARY_ELEMENTWISE_DESCRIPTOR(log, cpu, op::elementwise::unary::UnaryMode::Log)

#endif // __LOG_CPU_H__
