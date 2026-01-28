#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "bitwise_or_nvidia.cuh"

namespace op::bitwise_or::nvidia {

ELEMENTWISE_NVIDIA_IMPL_BINARY_INTEGRAL(bitwise_or)

} // namespace op::bitwise_or::nvidia
