#include "isfinite_cpu.h"
#include "../../../elementwise/cpu/elementwise_cpu_impl.h"

namespace op::isfinite::cpu {

ELEMENTWISE_CPU_IMPL_UNARY(isfinite)

} // namespace op::isfinite::cpu
