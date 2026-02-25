#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "ceil_nvidia.cuh"

namespace op::ceil::nvidia {

ELEMENTWISE_NVIDIA_IMPL_UNARY(ceil)

} // namespace op::ceil::nvidia
