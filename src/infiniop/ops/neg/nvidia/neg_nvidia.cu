#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "neg_nvidia.cuh"

namespace op::neg::nvidia {

ELEMENTWISE_NVIDIA_IMPL_UNARY(neg)

} // namespace op::neg::nvidia
