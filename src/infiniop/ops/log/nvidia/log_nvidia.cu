#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "log_nvidia.cuh"

namespace op::log::nvidia {

ELEMENTWISE_NVIDIA_IMPL_UNARY(log)

} // namespace op::log::nvidia
