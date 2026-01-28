#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "reciprocal_nvidia.cuh"

namespace op::reciprocal::nvidia {

ELEMENTWISE_NVIDIA_IMPL_UNARY(reciprocal)

} // namespace op::reciprocal::nvidia
