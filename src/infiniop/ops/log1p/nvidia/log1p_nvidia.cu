#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "log1p_nvidia.cuh"

namespace op::log1p::nvidia {

ELEMENTWISE_NVIDIA_IMPL_UNARY(log1p)

} // namespace op::log1p::nvidia
