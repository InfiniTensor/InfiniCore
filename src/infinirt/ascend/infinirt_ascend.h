#ifndef __INFINIRT_ASCEND_H__
#define __INFINIRT_ASCEND_H__
#include "../infinirt_impl.h"

namespace infinirt::ascend {
#ifdef ENABLE_ASCEND_API
infiniStatus_t init();
INFINIRT_DEVICE_API_IMPL
infiniStatus_t graphTaskGroupBegin(infinirtStream_t stream);
infiniStatus_t graphTaskGroupEnd(
    infinirtStream_t stream,
    infinirtGraphTaskGroup_t *handle);
infiniStatus_t graphTaskUpdateBegin(
    infinirtStream_t stream,
    infinirtGraphTaskGroup_t handle);
infiniStatus_t graphTaskUpdateEnd(infinirtStream_t stream);
#else
INFINIRT_DEVICE_API_NOOP
#endif
} // namespace infinirt::ascend

#endif // __INFINIRT_ASCEND_H__
