#ifndef INFINICCL_IMPL_H
#define INFINICCL_IMPL_H

#include "infiniccl.h"

struct InfinicclComm {
    infiniDevice_t device_type;
    int device_id; // the actual device ID, not rank number
    void *comm;    // the actual communicator
    int rank = 0;
    int world_size = 1;
    infinicclAllReduceBackend_t allreduce_backend = INFINICCL_ALLREDUCE_BACKEND_NCCL;
    void *custom_allreduce_context = nullptr;
};

#define INFINICCL_DEVICE_API(NAMSPACE, IMPL)               \
    namespace infiniccl::NAMSPACE {                        \
    infiniStatus_t commInitAll(                            \
        infinicclComm_t *comms,                            \
        int ndevice,                                       \
        const int *device_ids) IMPL;                       \
                                                           \
    infiniStatus_t commDestroy(infinicclComm_t comm) IMPL; \
                                                           \
    infiniStatus_t registerAllReduceBuffers(               \
        infinicclComm_t *comms,                            \
        int ndevice,                                       \
        void **buffers,                                    \
        size_t bytes) IMPL;                                \
                                                           \
    infiniStatus_t registerAllReduceBuffer(                \
        infinicclComm_t comm,                              \
        const char *key,                                   \
        void *buffer,                                      \
        size_t bytes) IMPL;                                \
                                                           \
    infiniStatus_t clearAllReduceBuffers(                  \
        infinicclComm_t *comms,                            \
        int ndevice) IMPL;                                 \
                                                           \
    infiniStatus_t groupStart(infinicclComm_t comm) IMPL;  \
                                                           \
    infiniStatus_t groupEnd(infinicclComm_t comm) IMPL;    \
                                                           \
    infiniStatus_t allReduce(                              \
        void *sendbuf,                                     \
        void *recvbuf,                                     \
        size_t count,                                      \
        infiniDtype_t datatype,                            \
        infinicclReduceOp_t op,                            \
        infinicclComm_t comm,                              \
        infinirtStream_t stream) IMPL;                     \
                                                           \
    infiniStatus_t allGather(                              \
        void *sendbuf,                                     \
        void *recvbuf,                                     \
        size_t send_count,                                 \
        infiniDtype_t datatype,                            \
        infinicclComm_t comm,                              \
        infinirtStream_t stream) IMPL;                     \
                                                           \
    infiniStatus_t allGatherV(                             \
        void *sendbuf,                                     \
        void *recvbuf,                                     \
        const size_t *recv_counts,                         \
        int nranks,                                        \
        infiniDtype_t datatype,                            \
        infinicclComm_t comm,                              \
        infinirtStream_t stream) IMPL;                     \
                                                           \
    infiniStatus_t reduceScatter(                          \
        void *sendbuf,                                     \
        void *recvbuf,                                     \
        size_t recv_count,                                 \
        infiniDtype_t datatype,                            \
        infinicclReduceOp_t op,                            \
        infinicclComm_t comm,                              \
        infinirtStream_t stream) IMPL;                     \
                                                           \
    infiniStatus_t reduceScatterV(                         \
        void *sendbuf,                                     \
        void *recvbuf,                                     \
        const size_t *send_counts,                         \
        int nranks,                                        \
        infiniDtype_t datatype,                            \
        infinicclReduceOp_t op,                            \
        infinicclComm_t comm,                              \
        infinirtStream_t stream) IMPL;                     \
    };

#define INFINICCL_DEVICE_API_IMPL(NAMSPACE) \
    INFINICCL_DEVICE_API(NAMSPACE, )

#define INFINICCL_DEVICE_API_NOOP(NAMSPACE) \
    INFINICCL_DEVICE_API(NAMSPACE, { return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED; })

#define INFINICCL_DEVICE_API_UNSUPPORTED_EP_COLLECTIVES() \
    infiniStatus_t groupStart(infinicclComm_t) {          \
        return INFINI_STATUS_NOT_IMPLEMENTED;             \
    }                                                     \
                                                          \
    infiniStatus_t groupEnd(infinicclComm_t) {            \
        return INFINI_STATUS_NOT_IMPLEMENTED;             \
    }                                                     \
                                                          \
    infiniStatus_t allGather(                             \
        void *,                                           \
        void *,                                           \
        size_t,                                           \
        infiniDtype_t,                                    \
        infinicclComm_t,                                  \
        infinirtStream_t) {                               \
        return INFINI_STATUS_NOT_IMPLEMENTED;             \
    }                                                     \
                                                          \
    infiniStatus_t allGatherV(                            \
        void *,                                           \
        void *,                                           \
        const size_t *,                                   \
        int,                                              \
        infiniDtype_t,                                    \
        infinicclComm_t,                                  \
        infinirtStream_t) {                               \
        return INFINI_STATUS_NOT_IMPLEMENTED;             \
    }                                                     \
                                                          \
    infiniStatus_t reduceScatter(                         \
        void *,                                           \
        void *,                                           \
        size_t,                                           \
        infiniDtype_t,                                    \
        infinicclReduceOp_t,                              \
        infinicclComm_t,                                  \
        infinirtStream_t) {                               \
        return INFINI_STATUS_NOT_IMPLEMENTED;             \
    }                                                     \
                                                          \
    infiniStatus_t reduceScatterV(                        \
        void *,                                           \
        void *,                                           \
        const size_t *,                                   \
        int,                                              \
        infiniDtype_t,                                    \
        infinicclReduceOp_t,                              \
        infinicclComm_t,                                  \
        infinirtStream_t) {                               \
        return INFINI_STATUS_NOT_IMPLEMENTED;             \
    }

#endif // INFINICCL_IMPL_H
