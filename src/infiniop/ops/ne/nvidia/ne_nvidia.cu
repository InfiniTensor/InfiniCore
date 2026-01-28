#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "ne_nvidia.cuh"

namespace op::ne::nvidia {

ELEMENTWISE_NVIDIA_IMPL_BINARY(ne)

} // namespace op::ne::nvidia
