#ifndef __SQUARE_CPU_H__
#define __SQUARE_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "../../../elementwise/unary.h"

UNARY_ELEMENTWISE_DESCRIPTOR(square, cpu, op::elementwise::unary::UnaryMode::Square)

#endif // __SQUARE_CPU_H__
