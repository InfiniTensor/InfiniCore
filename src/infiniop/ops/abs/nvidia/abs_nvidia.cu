#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "abs_nvidia.cuh"

namespace op::abs::nvidia {

ELEMENTWISE_NVIDIA_IMPL_UNARY(abs)

} // namespace op::abs::nvidia
