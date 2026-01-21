#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "square_nvidia.cuh"

namespace op::square::nvidia {

ELEMENTWISE_NVIDIA_IMPL_UNARY(square)

} // namespace op::square::nvidia
