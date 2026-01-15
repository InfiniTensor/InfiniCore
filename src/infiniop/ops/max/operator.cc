#include "../../operator_impl.h"
#include "infiniop/ops/max.h"

#ifdef ENABLE_CPU_API
#include "cpu/max_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API)
#include "nvidia/max_nvidia.cuh"
#endif

BINARY_OP_IMPL(max, Max)
