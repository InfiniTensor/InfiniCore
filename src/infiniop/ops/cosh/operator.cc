#include "../../operator_impl.h"
#include "infiniop/ops/cosh.h"

#ifdef ENABLE_CPU_API
#include "cpu/cosh_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API)
#include "nvidia/cosh_nvidia.cuh"
#endif

UNARY_OP_IMPL(cosh, Cosh)
