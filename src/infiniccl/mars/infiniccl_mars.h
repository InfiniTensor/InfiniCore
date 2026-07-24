#ifndef INFINICCL_MARS_H_
#define INFINICCL_MARS_H_

#include "../infiniccl_impl.h"

#if defined(ENABLE_MARS_API) && defined(ENABLE_CCL)
INFINICCL_DEVICE_API_IMPL(mars)
#else
INFINICCL_DEVICE_API_NOOP(mars)
#endif

#endif // INFINICCL_MARS_H_
