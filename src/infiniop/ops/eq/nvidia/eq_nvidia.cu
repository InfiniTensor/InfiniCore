#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "eq_nvidia.cuh"

namespace op::eq::nvidia {

ELEMENTWISE_NVIDIA_IMPL_BINARY(eq)

} // namespace op::eq::nvidia
