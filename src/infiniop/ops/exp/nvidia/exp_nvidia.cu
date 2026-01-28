#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "exp_nvidia.cuh"

namespace op::exp::nvidia {

ELEMENTWISE_NVIDIA_IMPL_UNARY_EXTENDED(exp)

} // namespace op::exp::nvidia
