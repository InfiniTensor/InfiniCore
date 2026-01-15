#include "../../operator_impl.h"
#include "infiniop/ops/floor.h"

#ifdef ENABLE_CPU_API
#include "cpu/floor_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API)
#include "nvidia/floor_nvidia.cuh"
#endif

UNARY_OP_IMPL(floor, Floor)
