#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "isfinite_nvidia.cuh"

namespace op::isfinite::nvidia {

ELEMENTWISE_NVIDIA_IMPL_UNARY(isfinite)

} // namespace op::isfinite::nvidia
