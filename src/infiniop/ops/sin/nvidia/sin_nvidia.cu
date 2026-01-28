#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "sin_nvidia.cuh"

namespace op::sin::nvidia {

ELEMENTWISE_NVIDIA_IMPL_UNARY(sin)

} // namespace op::sin::nvidia
