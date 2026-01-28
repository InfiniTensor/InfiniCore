#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "lt_nvidia.cuh"

namespace op::lt::nvidia {

ELEMENTWISE_NVIDIA_IMPL_BINARY(lt)

} // namespace op::lt::nvidia
