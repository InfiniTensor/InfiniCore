#include "../../operator_impl.h"
#include "infiniop/ops/sign.h"

#ifdef ENABLE_CPU_API
#include "cpu/sign_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API)
#include "nvidia/sign_nvidia.cuh"
#endif

UNARY_OP_IMPL(sign, Sign)
