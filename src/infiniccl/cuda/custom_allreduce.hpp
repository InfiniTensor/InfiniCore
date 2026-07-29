#ifndef INFINICCL_CUDA_CUSTOM_ALLREDUCE_HPP_
#define INFINICCL_CUDA_CUSTOM_ALLREDUCE_HPP_

#include "../infiniccl_impl.h"

namespace infiniccl::cuda {

struct CustomAllReduceContext;

struct CustomAllReduceCheckResult {
    bool supported = false;
    const char *reason = nullptr;
};

CustomAllReduceContext *createCustomAllReduceContext(
    int rank,
    int world_size,
    int device_id,
    const int *device_ids);

void initializeCustomAllReduceContexts(
    infinicclComm_t *comms,
    int ndevice,
    const int *device_ids);

void destroyCustomAllReduceContext(CustomAllReduceContext *ctx);

infiniStatus_t registerCustomAllReduceBuffers(
    infinicclComm_t *comms,
    int ndevice,
    void **buffers,
    size_t bytes);

infiniStatus_t registerCustomAllReduceBuffer(
    infinicclComm_t comm,
    const char *key,
    void *buffer,
    size_t bytes);

infiniStatus_t clearCustomAllReduceBuffers(
    infinicclComm_t *comms,
    int ndevice);

CustomAllReduceCheckResult canUseCustomAllReduce(
    const CustomAllReduceContext *ctx,
    size_t count,
    infiniDtype_t datatype,
    infinicclReduceOp_t op);

infiniStatus_t tryCustomAllReduce(
    CustomAllReduceContext *ctx,
    void *sendbuf,
    void *recvbuf,
    size_t count,
    infiniDtype_t datatype,
    infinicclReduceOp_t op,
    infinirtStream_t stream,
    bool *handled);

} // namespace infiniccl::cuda

#endif // INFINICCL_CUDA_CUSTOM_ALLREDUCE_HPP_