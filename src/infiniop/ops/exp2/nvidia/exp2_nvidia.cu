#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "exp2_nvidia.cuh"

namespace op::exp2::nvidia {

ELEMENTWISE_NVIDIA_IMPL_UNARY_EXTENDED(exp2)

} // namespace op::exp2::nvidia
