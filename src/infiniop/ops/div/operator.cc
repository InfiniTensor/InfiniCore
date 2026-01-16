#include "../../operator_impl.h"
#include "infiniop/ops/div.h"

#ifdef ENABLE_CPU_API
#include "cpu/div_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API)
#include "nvidia/div_nvidia.cuh"
#endif

BINARY_OP_IMPL(div, Div)
