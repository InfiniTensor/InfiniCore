#include "../../operator_impl.h"
#include "infiniop/ops/acosh.h"

#ifdef ENABLE_CPU_API
#include "cpu/acosh_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API)
#include "nvidia/acosh_nvidia.cuh"
#endif

UNARY_OP_IMPL(acosh, Acosh)
