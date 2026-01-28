#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "min_nvidia.cuh"

namespace op::min::nvidia {

ELEMENTWISE_NVIDIA_IMPL_BINARY(min)

} // namespace op::min::nvidia
