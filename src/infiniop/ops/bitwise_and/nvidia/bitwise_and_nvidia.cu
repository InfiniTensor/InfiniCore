#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "bitwise_and_nvidia.cuh"

namespace op::bitwise_and::nvidia {

ELEMENTWISE_NVIDIA_IMPL_BINARY_INTEGRAL(bitwise_and)

} // namespace op::bitwise_and::nvidia
