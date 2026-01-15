#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "pow_nvidia.cuh"

namespace op::pow::nvidia {

ELEMENTWISE_NVIDIA_IMPL_BINARY(pow)

} // namespace op::pow::nvidia
