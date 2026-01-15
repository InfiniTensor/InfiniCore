#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "tanh_nvidia.cuh"

namespace op::tanh::nvidia {

ELEMENTWISE_NVIDIA_IMPL_UNARY(tanh)

} // namespace op::tanh::nvidia
