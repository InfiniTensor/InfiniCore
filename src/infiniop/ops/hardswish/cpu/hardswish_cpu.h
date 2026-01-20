#ifndef __HARDSWISH_CPU_H__
#define __HARDSWISH_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "../../../elementwise/unary.h"

UNARY_ELEMENTWISE_DESCRIPTOR(hardswish, cpu, op::elementwise::unary::UnaryMode::Hardswish)

#endif // __HARDSWISH_CPU_H__
