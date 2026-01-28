#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "isinf_nvidia.cuh"

namespace op::isinf::nvidia {

ELEMENTWISE_NVIDIA_IMPL_UNARY(isinf)

} // namespace op::isinf::nvidia
