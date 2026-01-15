#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "atan_nvidia.cuh"

namespace op::atan::nvidia {

ELEMENTWISE_NVIDIA_IMPL_UNARY(atan)

} // namespace op::atan::nvidia
