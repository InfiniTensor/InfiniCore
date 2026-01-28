#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "cosh_nvidia.cuh"

namespace op::cosh::nvidia {

ELEMENTWISE_NVIDIA_IMPL_UNARY(cosh)

} // namespace op::cosh::nvidia
