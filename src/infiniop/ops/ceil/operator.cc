#include "../../operator_impl.h"
#include "infiniop/ops/ceil.h"

#ifdef ENABLE_CPU_API
#include "cpu/ceil_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API)
#include "nvidia/ceil_nvidia.cuh"
#endif

UNARY_OP_IMPL(ceil, Ceil)
