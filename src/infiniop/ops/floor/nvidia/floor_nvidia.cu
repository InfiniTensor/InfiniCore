#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "floor_nvidia.cuh"

namespace op::floor::nvidia {

ELEMENTWISE_NVIDIA_IMPL_UNARY(floor)

} // namespace op::floor::nvidia
