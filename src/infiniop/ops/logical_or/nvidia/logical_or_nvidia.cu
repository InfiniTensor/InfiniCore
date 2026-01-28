#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "logical_or_nvidia.cuh"

namespace op::logical_or::nvidia {

ELEMENTWISE_NVIDIA_IMPL_BINARY(logical_or)

} // namespace op::logical_or::nvidia
