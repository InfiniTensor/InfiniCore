#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "atanh_nvidia.cuh"

namespace op::atanh::nvidia {

ELEMENTWISE_NVIDIA_IMPL_UNARY(atanh)

} // namespace op::atanh::nvidia
