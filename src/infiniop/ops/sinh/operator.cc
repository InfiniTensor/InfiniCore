#include "../../operator_impl.h"
#include "infiniop/ops/sinh.h"

#ifdef ENABLE_CPU_API
#include "cpu/sinh_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API)
#include "nvidia/sinh_nvidia.cuh"
#endif

UNARY_OP_IMPL(sinh, Sinh)
