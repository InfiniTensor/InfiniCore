#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "asinh_nvidia.cuh"

namespace op::asinh::nvidia {

ELEMENTWISE_NVIDIA_IMPL_UNARY(asinh)

} // namespace op::asinh::nvidia
