#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "logical_and_nvidia.cuh"

namespace op::logical_and::nvidia {

ELEMENTWISE_NVIDIA_IMPL_BINARY(logical_and)

} // namespace op::logical_and::nvidia
