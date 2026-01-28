#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "floor_divide_nvidia.cuh"

namespace op::floor_divide::nvidia {

ELEMENTWISE_NVIDIA_IMPL_BINARY(floor_divide)

} // namespace op::floor_divide::nvidia
