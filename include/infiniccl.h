#ifndef __INFINICCL_API_H__
#define __INFINICCL_API_H__

#include "infinirt.h"

typedef enum {
    INFINICCL_SUM = 0,
    INFINICCL_PROD = 1,
    INFINICCL_MAX = 2,
    INFINICCL_MIN = 3,
    INFINICCL_AVG = 4,
} infinicclReduceOp_t;

typedef enum {
    INFINICCL_ALLREDUCE_BACKEND_AUTO = 0,
    INFINICCL_ALLREDUCE_BACKEND_NCCL = 1,
    INFINICCL_ALLREDUCE_BACKEND_CUSTOM = 2,
} infinicclAllReduceBackend_t;

struct InfinicclComm;

typedef struct InfinicclComm *infinicclComm_t;

__INFINI_C __export infiniStatus_t infinicclCommInitAll(
    infiniDevice_t device_type,
    infinicclComm_t *comms,
    int ndevice,
    const int *device_ids);

__INFINI_C __export infiniStatus_t infinicclCommDestroy(infinicclComm_t comm);

__INFINI_C __export infiniStatus_t infinicclCommSetAllReduceBackend(
    infinicclComm_t comm,
    infinicclAllReduceBackend_t backend);

__INFINI_C __export infiniStatus_t infinicclCommGetAllReduceBackend(
    infinicclComm_t comm,
    infinicclAllReduceBackend_t *backend);

__INFINI_C __export infiniStatus_t infinicclCommRegisterAllReduceBuffers(
    infinicclComm_t *comms,
    int ndevice,
    void **buffers,
    size_t bytes);

__INFINI_C __export infiniStatus_t infinicclCommRegisterAllReduceBuffer(
    infinicclComm_t comm,
    const char *key,
    void *buffer,
    size_t bytes);

__INFINI_C __export infiniStatus_t infinicclCommClearAllReduceBuffers(
    infinicclComm_t *comms,
    int ndevice);

__INFINI_C __export infiniStatus_t infinicclGroupStart(infinicclComm_t comm);

__INFINI_C __export infiniStatus_t infinicclGroupEnd(infinicclComm_t comm);

__INFINI_C __export infiniStatus_t infinicclAllReduce(
    void *sendbuf,
    void *recvbuf,
    size_t count,
    infiniDtype_t dataype,
    infinicclReduceOp_t op,
    infinicclComm_t comm,
    infinirtStream_t stream);

__INFINI_C __export infiniStatus_t infinicclAllGather(
    void *sendbuf,
    void *recvbuf,
    size_t send_count,
    infiniDtype_t dataype,
    infinicclComm_t comm,
    infinirtStream_t stream);

__INFINI_C __export infiniStatus_t infinicclAllGatherV(
    void *sendbuf,
    void *recvbuf,
    const size_t *recv_counts,
    int nranks,
    infiniDtype_t dataype,
    infinicclComm_t comm,
    infinirtStream_t stream);

__INFINI_C __export infiniStatus_t infinicclReduceScatter(
    void *sendbuf,
    void *recvbuf,
    size_t recv_count,
    infiniDtype_t dataype,
    infinicclReduceOp_t op,
    infinicclComm_t comm,
    infinirtStream_t stream);

__INFINI_C __export infiniStatus_t infinicclReduceScatterV(
    void *sendbuf,
    void *recvbuf,
    const size_t *send_counts,
    int nranks,
    infiniDtype_t dataype,
    infinicclReduceOp_t op,
    infinicclComm_t comm,
    infinirtStream_t stream);

#endif
