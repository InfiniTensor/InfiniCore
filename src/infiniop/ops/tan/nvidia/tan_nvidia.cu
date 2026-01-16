#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "tan_nvidia.cuh"

namespace op::tan::nvidia {

ELEMENTWISE_NVIDIA_IMPL_UNARY(tan)

} // namespace op::tan::nvidia
