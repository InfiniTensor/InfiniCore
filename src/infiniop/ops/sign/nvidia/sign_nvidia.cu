#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "sign_nvidia.cuh"

namespace op::sign::nvidia {

ELEMENTWISE_NVIDIA_IMPL_UNARY(sign)

} // namespace op::sign::nvidia
