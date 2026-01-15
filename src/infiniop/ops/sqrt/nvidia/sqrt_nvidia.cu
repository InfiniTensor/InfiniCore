#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "sqrt_nvidia.cuh"

namespace op::sqrt::nvidia {

ELEMENTWISE_NVIDIA_IMPL_UNARY(sqrt)

} // namespace op::sqrt::nvidia
