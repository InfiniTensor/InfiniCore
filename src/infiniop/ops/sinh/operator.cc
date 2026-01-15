#include "../../operator_impl.h"
#include "infiniop/ops/sinh.h"

#ifdef ENABLE_CPU_API
#include "cpu/sinh_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_QY_API)
#include "nvidia/sinh_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/sinh_metax.h"
#endif
#ifdef ENABLE_MOORE_API
#include "moore/sinh_moore.h"
#endif

UNARY_OP_IMPL(sinh, Sinh)
