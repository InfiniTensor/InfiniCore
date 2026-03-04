#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "acos_nvidia.cuh"

namespace op::acos::nvidia {

ELEMENTWISE_NVIDIA_IMPL_UNARY(acos)

} // namespace op::acos::nvidia
