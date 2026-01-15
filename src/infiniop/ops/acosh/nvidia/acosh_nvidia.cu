#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "acosh_nvidia.cuh"

namespace op::acosh::nvidia {

ELEMENTWISE_NVIDIA_IMPL_UNARY(acosh)

} // namespace op::acosh::nvidia
