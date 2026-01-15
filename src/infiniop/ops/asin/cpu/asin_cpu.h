#ifndef __ASIN_CPU_H__
#define __ASIN_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "../../../elementwise/unary.h"

UNARY_ELEMENTWISE_DESCRIPTOR(asin, cpu, op::elementwise::unary::UnaryMode::Asin)

#endif // __ASIN_CPU_H__
