#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "round_nvidia.cuh"

namespace op::round::nvidia {

ELEMENTWISE_NVIDIA_IMPL_UNARY(round)

} // namespace op::round::nvidia
