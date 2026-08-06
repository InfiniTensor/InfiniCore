#pragma once

#include "../infiniccl_impl.h"

namespace infiniccl::hygon {

void customAllReduceInitAll(
    infinicclComm_t *comms,
    int ndevice,
    const int *device_ids);

void customAllReduceDestroy(infinicclComm_t comm);

bool customAllReduce(
    void *sendbuf,
    void *recvbuf,
    size_t count,
    infiniDtype_t datatype,
    infinicclReduceOp_t op,
    infinicclComm_t comm,
    infinirtStream_t stream);

} // namespace infiniccl::hygon
