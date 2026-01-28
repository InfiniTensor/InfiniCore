#ifndef __SINC_CPU_H__
#define __SINC_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "../../../elementwise/unary.h"

UNARY_ELEMENTWISE_DESCRIPTOR(sinc, cpu, op::elementwise::unary::UnaryMode::Sinc)

#endif // __SINC_CPU_H__
