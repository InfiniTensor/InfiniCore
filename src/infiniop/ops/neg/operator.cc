#include "../../operator_impl.h"
#include "infiniop/ops/neg.h"

#ifdef ENABLE_CPU_API
#include "cpu/neg_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API)
#include "nvidia/neg_nvidia.cuh"
#endif

UNARY_OP_IMPL(neg, Neg)
