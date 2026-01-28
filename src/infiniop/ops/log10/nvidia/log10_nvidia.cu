#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "log10_nvidia.cuh"

namespace op::log10::nvidia {

ELEMENTWISE_NVIDIA_IMPL_UNARY_EXTENDED(log10)

} // namespace op::log10::nvidia
