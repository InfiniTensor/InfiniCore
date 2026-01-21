#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "bitwise_left_shift_nvidia.cuh"

namespace op::bitwise_left_shift::nvidia {

ELEMENTWISE_NVIDIA_IMPL_BINARY_INTEGRAL(bitwise_left_shift)

} // namespace op::bitwise_left_shift::nvidia
