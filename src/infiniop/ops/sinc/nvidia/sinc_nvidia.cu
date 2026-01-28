#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "sinc_nvidia.cuh"

namespace op::sinc::nvidia {

ELEMENTWISE_NVIDIA_IMPL_UNARY(sinc)

} // namespace op::sinc::nvidia
