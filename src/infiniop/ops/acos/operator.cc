#include "../../operator_impl.h"
#include "infiniop/ops/acos.h"

#ifdef ENABLE_CPU_API
#include "cpu/acos_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API)
#include "nvidia/acos_nvidia.cuh"
#endif

UNARY_OP_IMPL(acos, Acos)
