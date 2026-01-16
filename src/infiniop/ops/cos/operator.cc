#include "../../operator_impl.h"
#include "infiniop/ops/cos.h"

#ifdef ENABLE_CPU_API
#include "cpu/cos_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API)
#include "nvidia/cos_nvidia.cuh"
#endif

UNARY_OP_IMPL(cos, Cos)
