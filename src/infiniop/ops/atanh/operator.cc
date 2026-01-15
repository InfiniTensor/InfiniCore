#include "../../operator_impl.h"
#include "infiniop/ops/atanh.h"

#ifdef ENABLE_CPU_API
#include "cpu/atanh_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API)
#include "nvidia/atanh_nvidia.cuh"
#endif

UNARY_OP_IMPL(atanh, Atanh)
