#ifndef __INFINIOP_VAR_CPU_H__
#define __INFINIOP_VAR_CPU_H__

// #include "../../../elementwise/cpu/elementwise_cpu.h"  elementwise 的实现也是一个一个地按下标访问
// 后续需要优化访存、divergence、memory coalescing、vectorization、
#include "../var_desc.h"

// namespace op::var::cpu {
// }

DESCRIPTOR(cpu);


#endif // __INFINIOP_VAR_CPU_H__
