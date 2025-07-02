#ifndef INFINICCL_MUSA_H_
#define INFINICCL_MUSA_H_

#include "../infiniccl_impl.h"

#if defined(ENABLE_MOORE_API) && defined(ENABLE_CCL)
INFINICCL_DEVICE_API_IMPL(musa)
#else
INFINICCL_DEVICE_API_NOOP(musa)
#endif

#endif /* INFINICCL_MUSA_H_ */
