#include "../../operator_impl.h"
#include "infiniop/ops/sqrt.h"

#ifdef ENABLE_CPU_API
#include "cpu/sqrt_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API)
#include "nvidia/sqrt_nvidia.cuh"
#endif

UNARY_OP_IMPL(sqrt, Sqrt)
