#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "cos_nvidia.cuh"

namespace op::cos::nvidia {

ELEMENTWISE_NVIDIA_IMPL_UNARY(cos)

} // namespace op::cos::nvidia
