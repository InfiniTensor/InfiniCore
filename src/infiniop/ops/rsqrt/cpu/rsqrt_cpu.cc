#include "rsqrt_cpu.h"
#include "../../../elementwise/cpu/elementwise_cpu_impl.h"

namespace op::rsqrt::cpu {

ELEMENTWISE_CPU_IMPL_UNARY(rsqrt)

} // namespace op::rsqrt::cpu
