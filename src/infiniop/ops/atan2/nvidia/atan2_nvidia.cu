#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "atan2_nvidia.cuh"

namespace op::atan2::nvidia {

ELEMENTWISE_NVIDIA_IMPL_BINARY(atan2)

} // namespace op::atan2::nvidia
