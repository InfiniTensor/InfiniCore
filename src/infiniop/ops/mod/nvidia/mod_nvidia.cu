#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "mod_nvidia.cuh"

namespace op::mod::nvidia {

ELEMENTWISE_NVIDIA_IMPL_BINARY(mod)

} // namespace op::mod::nvidia
