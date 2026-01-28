#include "isnan_cpu.h"
#include "../../../elementwise/cpu/elementwise_cpu_impl.h"

namespace op::isnan::cpu {

ELEMENTWISE_CPU_IMPL_UNARY(isnan)

} // namespace op::isnan::cpu
