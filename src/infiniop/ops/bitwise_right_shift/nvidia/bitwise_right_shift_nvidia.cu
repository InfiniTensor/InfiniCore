#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "bitwise_right_shift_nvidia.cuh"

namespace op::bitwise_right_shift::nvidia {

ELEMENTWISE_NVIDIA_IMPL_BINARY_INTEGRAL(bitwise_right_shift)

} // namespace op::bitwise_right_shift::nvidia
