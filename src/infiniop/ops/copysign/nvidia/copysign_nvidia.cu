#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "copysign_nvidia.cuh"

namespace op::copysign::nvidia {

ELEMENTWISE_NVIDIA_IMPL_BINARY(copysign)

} // namespace op::copysign::nvidia
