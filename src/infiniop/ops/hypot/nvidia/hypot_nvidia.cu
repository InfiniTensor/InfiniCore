#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "hypot_nvidia.cuh"

namespace op::hypot::nvidia {

ELEMENTWISE_NVIDIA_IMPL_BINARY(hypot)

} // namespace op::hypot::nvidia
