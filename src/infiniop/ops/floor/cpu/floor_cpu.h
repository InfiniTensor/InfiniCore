#ifndef __FLOOR_CPU_H__
#define __FLOOR_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "../../../elementwise/unary.h"

UNARY_ELEMENTWISE_DESCRIPTOR(floor, cpu, op::elementwise::unary::UnaryMode::Floor)

#endif // __FLOOR_CPU_H__
