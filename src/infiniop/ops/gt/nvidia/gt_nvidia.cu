#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "gt_nvidia.cuh"

namespace op::gt::nvidia {

ELEMENTWISE_NVIDIA_IMPL_BINARY(gt)

} // namespace op::gt::nvidia
