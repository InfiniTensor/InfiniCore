#ifndef INFINICCL_ASCEND_H_
#define INFINICCL_ASCEND_H_

#include "../infiniccl_impl.h"

#if defined(ENABLE_ASCEND_API) && defined(ENABLE_CCL)
namespace infiniccl::ascend {
infiniStatus_t getCommName(
    infinicclComm_t comm,
    char *comm_name,
    size_t comm_name_size);
}
INFINICCL_DEVICE_API_IMPL(ascend)
#else
namespace infiniccl::ascend {
inline infiniStatus_t getCommName(
    infinicclComm_t,
    char *,
    size_t) {
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}
} // namespace infiniccl::ascend
INFINICCL_DEVICE_API_NOOP(ascend)
#endif

#endif /* INFINICCL_ASCEND_H_ */
