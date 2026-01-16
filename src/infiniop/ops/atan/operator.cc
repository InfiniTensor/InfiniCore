#include "../../operator_impl.h"
#include "infiniop/ops/unary_ops_api.h"

#ifdef ENABLE_CPU_API
#include "cpu/atan_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API)
#include "nvidia/atan_nvidia.cuh"
#endif

UNARY_OP_IMPL(atan, Atan)
