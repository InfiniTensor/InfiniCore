#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "erf_nvidia.cuh"

namespace op::erf::nvidia {

ELEMENTWISE_NVIDIA_IMPL_UNARY(erf)

} // namespace op::erf::nvidia
