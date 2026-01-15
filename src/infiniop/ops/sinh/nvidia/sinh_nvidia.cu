#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "sinh_nvidia.cuh"

namespace op::sinh::nvidia {

ELEMENTWISE_NVIDIA_IMPL_UNARY(sinh)

} // namespace op::sinh::nvidia
