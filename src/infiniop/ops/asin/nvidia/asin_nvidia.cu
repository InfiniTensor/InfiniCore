#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "asin_nvidia.cuh"

namespace op::asin::nvidia {

ELEMENTWISE_NVIDIA_IMPL_UNARY(asin)

} // namespace op::asin::nvidia
