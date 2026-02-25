#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "div_nvidia.cuh"

namespace op::div::nvidia {

ELEMENTWISE_NVIDIA_IMPL_BINARY(div)

} // namespace op::div::nvidia
