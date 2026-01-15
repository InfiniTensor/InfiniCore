#include "../../operator_impl.h"
#include "infiniop/ops/erf.h"

#ifdef ENABLE_CPU_API
#include "cpu/erf_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_QY_API)
#include "nvidia/erf_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/erf_metax.h"
#endif
#ifdef ENABLE_MOORE_API
#include "moore/erf_moore.h"
#endif

UNARY_OP_IMPL(erf, Erf)
