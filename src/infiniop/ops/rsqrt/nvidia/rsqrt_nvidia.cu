#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "rsqrt_nvidia.cuh"

namespace op::rsqrt::nvidia {

ELEMENTWISE_NVIDIA_IMPL_UNARY(rsqrt)

} // namespace op::rsqrt::nvidia
