#include "../../operator_impl.h"
#include "infiniop/ops/binary_ops_api.h"

#ifdef ENABLE_CPU_API
#include "cpu/min_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API)
#include "nvidia/min_nvidia.cuh"
#endif

BINARY_OP_IMPL(min, Min)
