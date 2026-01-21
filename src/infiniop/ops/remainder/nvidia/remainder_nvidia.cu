#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "remainder_nvidia.cuh"

namespace op::remainder::nvidia {

ELEMENTWISE_NVIDIA_IMPL_BINARY(remainder)

} // namespace op::remainder::nvidia
