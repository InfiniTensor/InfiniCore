#ifndef INFINICCL_IMPL_H
#define INFINICCL_IMPL_H

#include "infiniccl.h"

struct InfinicclComm {
    infiniDevice_t device_type;
    int device_id; // the actual device ID, not rank number
    void *comm;    // the actual communicator
    int rank = 0;
    int world_size = 1;
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

#endif // INFINICCL_IMPL_H
