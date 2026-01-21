#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "bitwise_xor_nvidia.cuh"

namespace op::bitwise_xor::nvidia {

ELEMENTWISE_NVIDIA_IMPL_BINARY_INTEGRAL(bitwise_xor)

} // namespace op::bitwise_xor::nvidia
