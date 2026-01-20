#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "hardswish_nvidia.cuh"

namespace op::hardswish::nvidia {

ELEMENTWISE_NVIDIA_IMPL_UNARY_EXTENDED(hardswish)

} // namespace op::hardswish::nvidia
