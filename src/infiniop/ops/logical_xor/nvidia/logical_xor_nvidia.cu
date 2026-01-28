#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "logical_xor_nvidia.cuh"

namespace op::logical_xor::nvidia {

ELEMENTWISE_NVIDIA_IMPL_BINARY(logical_xor)

} // namespace op::logical_xor::nvidia
