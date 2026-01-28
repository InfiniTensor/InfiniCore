#include "../../operator_impl.h"
#include "infiniop/ops/unary_ops_api.h"

#ifdef ENABLE_CPU_API
#include "cpu/rsqrt_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API)
#include "nvidia/rsqrt_nvidia.cuh"
#endif

UNARY_OP_IMPL(rsqrt, Rsqrt)
