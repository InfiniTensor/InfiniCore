#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "fmin_nvidia.cuh"

namespace op::fmin::nvidia {

ELEMENTWISE_NVIDIA_IMPL_BINARY(fmin)

} // namespace op::fmin::nvidia
