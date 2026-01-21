#include "../../operator_impl.h"
#include "infiniop/ops/binary_ops_api.h"

#ifdef ENABLE_CPU_API
#include "cpu/bitwise_right_shift_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API)
#include "nvidia/bitwise_right_shift_nvidia.cuh"
#endif

BINARY_OP_IMPL(bitwise_right_shift, BitwiseRightShift)
