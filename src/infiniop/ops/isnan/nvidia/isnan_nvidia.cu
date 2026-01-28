#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "isnan_nvidia.cuh"

namespace op::isnan::nvidia {

ELEMENTWISE_NVIDIA_IMPL_UNARY(isnan)

} // namespace op::isnan::nvidia
