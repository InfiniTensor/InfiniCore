#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "fmax_nvidia.cuh"

namespace op::fmax::nvidia {

ELEMENTWISE_NVIDIA_IMPL_BINARY(fmax)

} // namespace op::fmax::nvidia
