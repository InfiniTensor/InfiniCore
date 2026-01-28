#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "log2_nvidia.cuh"

namespace op::log2::nvidia {

ELEMENTWISE_NVIDIA_IMPL_UNARY_EXTENDED(log2)

} // namespace op::log2::nvidia
