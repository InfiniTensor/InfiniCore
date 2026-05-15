#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "max_nvidia.cuh"

namespace op::max::nvidia {

ELEMENTWISE_NVIDIA_IMPL_BINARY(max)

} // namespace op::max::nvidia
