#include "infiniccl.h"

#include "./ascend/infiniccl_ascend.h"
#include "./cambricon/infiniccl_cambricon.h"
#include "./cuda/infiniccl_cuda.h"
#include "./kunlun/infiniccl_kunlun.h"
#include "./metax/infiniccl_metax.h"
#include "./moore/infiniccl_moore.h"

__INFINI_C infiniStatus_t infinicclCommInitAll(
    infiniDevice_t device_type,
    infinicclComm_t *comms,
    int ndevice,
    const int *device_ids) {

#define COMM_INIT_ALL(CASE_, NAMESPACE_) \
    case CASE_:                          \
        return infiniccl::NAMESPACE_::commInitAll(comms, ndevice, device_ids)

    switch (device_type) {
        COMM_INIT_ALL(INFINI_DEVICE_NVIDIA, cuda);
        COMM_INIT_ALL(INFINI_DEVICE_ILUVATAR, cuda);
        COMM_INIT_ALL(INFINI_DEVICE_QY, cuda);
        COMM_INIT_ALL(INFINI_DEVICE_HYGON, cuda);
        COMM_INIT_ALL(INFINI_DEVICE_ASCEND, ascend);
        COMM_INIT_ALL(INFINI_DEVICE_CAMBRICON, cambricon);
        COMM_INIT_ALL(INFINI_DEVICE_METAX, metax);
        COMM_INIT_ALL(INFINI_DEVICE_MOORE, moore);
        COMM_INIT_ALL(INFINI_DEVICE_KUNLUN, kunlun);
        COMM_INIT_ALL(INFINI_DEVICE_ALI, cuda);
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef COMM_INIT_ALL
}

__INFINI_C infiniStatus_t infinicclCommDestroy(infinicclComm_t comm) {
    if (comm == nullptr) {
        return INFINI_STATUS_SUCCESS;
    }

#define COMM_DESTROY(CASE_, NAMESPACE_) \
    case CASE_:                         \
        return infiniccl::NAMESPACE_::commDestroy(comm)

    switch (comm->device_type) {
        COMM_DESTROY(INFINI_DEVICE_NVIDIA, cuda);
        COMM_DESTROY(INFINI_DEVICE_ILUVATAR, cuda);
        COMM_DESTROY(INFINI_DEVICE_QY, cuda);
        COMM_DESTROY(INFINI_DEVICE_HYGON, cuda);
        COMM_DESTROY(INFINI_DEVICE_ASCEND, ascend);
        COMM_DESTROY(INFINI_DEVICE_CAMBRICON, cambricon);
        COMM_DESTROY(INFINI_DEVICE_METAX, metax);
        COMM_DESTROY(INFINI_DEVICE_MOORE, moore);
        COMM_DESTROY(INFINI_DEVICE_KUNLUN, kunlun);
        COMM_DESTROY(INFINI_DEVICE_ALI, cuda);
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef COMM_DESTROY
}

__INFINI_C infiniStatus_t infinicclGroupStart(infinicclComm_t comm) {
    if (comm == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }

#define GROUP_START(CASE_, NAMESPACE_) \
    case CASE_:                        \
        return infiniccl::NAMESPACE_::groupStart(comm)

    switch (comm->device_type) {
        GROUP_START(INFINI_DEVICE_NVIDIA, cuda);
        GROUP_START(INFINI_DEVICE_ILUVATAR, cuda);
        GROUP_START(INFINI_DEVICE_QY, cuda);
        GROUP_START(INFINI_DEVICE_HYGON, cuda);
        GROUP_START(INFINI_DEVICE_ASCEND, ascend);
        GROUP_START(INFINI_DEVICE_CAMBRICON, cambricon);
        GROUP_START(INFINI_DEVICE_METAX, metax);
        GROUP_START(INFINI_DEVICE_MOORE, moore);
        GROUP_START(INFINI_DEVICE_KUNLUN, kunlun);
        GROUP_START(INFINI_DEVICE_ALI, cuda);
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef GROUP_START
}

__INFINI_C infiniStatus_t infinicclGroupEnd(infinicclComm_t comm) {
    if (comm == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }

#define GROUP_END(CASE_, NAMESPACE_) \
    case CASE_:                      \
        return infiniccl::NAMESPACE_::groupEnd(comm)

    switch (comm->device_type) {
        GROUP_END(INFINI_DEVICE_NVIDIA, cuda);
        GROUP_END(INFINI_DEVICE_ILUVATAR, cuda);
        GROUP_END(INFINI_DEVICE_QY, cuda);
        GROUP_END(INFINI_DEVICE_HYGON, cuda);
        GROUP_END(INFINI_DEVICE_ASCEND, ascend);
        GROUP_END(INFINI_DEVICE_CAMBRICON, cambricon);
        GROUP_END(INFINI_DEVICE_METAX, metax);
        GROUP_END(INFINI_DEVICE_MOORE, moore);
        GROUP_END(INFINI_DEVICE_KUNLUN, kunlun);
        GROUP_END(INFINI_DEVICE_ALI, cuda);
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef GROUP_END
}

__INFINI_C infiniStatus_t infinicclAllReduce(
    void *sendbuf,
    void *recvbuf,
    size_t count,
    infiniDtype_t dataype,
    infinicclReduceOp_t op,
    infinicclComm_t comm,
    infinirtStream_t stream) {

    if (comm == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }

#define ALL_REDUCE(CASE_, NAMESPACE_) \
    case CASE_:                       \
        return infiniccl::NAMESPACE_::allReduce(sendbuf, recvbuf, count, dataype, op, comm, stream)

    switch (comm->device_type) {
        ALL_REDUCE(INFINI_DEVICE_NVIDIA, cuda);
        ALL_REDUCE(INFINI_DEVICE_ILUVATAR, cuda);
        ALL_REDUCE(INFINI_DEVICE_QY, cuda);
        ALL_REDUCE(INFINI_DEVICE_HYGON, cuda);
        ALL_REDUCE(INFINI_DEVICE_ASCEND, ascend);
        ALL_REDUCE(INFINI_DEVICE_CAMBRICON, cambricon);
        ALL_REDUCE(INFINI_DEVICE_METAX, metax);
        ALL_REDUCE(INFINI_DEVICE_MOORE, moore);
        ALL_REDUCE(INFINI_DEVICE_KUNLUN, kunlun);
        ALL_REDUCE(INFINI_DEVICE_ALI, cuda);

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef ALL_REDUCE
}

__INFINI_C infiniStatus_t infinicclAllGather(
    void *sendbuf,
    void *recvbuf,
    size_t send_count,
    infiniDtype_t dataype,
    infinicclComm_t comm,
    infinirtStream_t stream) {

    if (comm == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }

#define ALL_GATHER(CASE_, NAMESPACE_) \
    case CASE_:                       \
        return infiniccl::NAMESPACE_::allGather(sendbuf, recvbuf, send_count, dataype, comm, stream)

    switch (comm->device_type) {
        ALL_GATHER(INFINI_DEVICE_NVIDIA, cuda);
        ALL_GATHER(INFINI_DEVICE_ILUVATAR, cuda);
        ALL_GATHER(INFINI_DEVICE_QY, cuda);
        ALL_GATHER(INFINI_DEVICE_HYGON, cuda);
        ALL_GATHER(INFINI_DEVICE_ASCEND, ascend);
        ALL_GATHER(INFINI_DEVICE_CAMBRICON, cambricon);
        ALL_GATHER(INFINI_DEVICE_METAX, metax);
        ALL_GATHER(INFINI_DEVICE_MOORE, moore);
        ALL_GATHER(INFINI_DEVICE_KUNLUN, kunlun);
        ALL_GATHER(INFINI_DEVICE_ALI, cuda);

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef ALL_GATHER
}

__INFINI_C infiniStatus_t infinicclAllGatherV(
    void *sendbuf,
    void *recvbuf,
    const size_t *recv_counts,
    int nranks,
    infiniDtype_t dataype,
    infinicclComm_t comm,
    infinirtStream_t stream) {

    if (comm == nullptr || recv_counts == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }

#define ALL_GATHER_V(CASE_, NAMESPACE_) \
    case CASE_:                         \
        return infiniccl::NAMESPACE_::allGatherV(sendbuf, recvbuf, recv_counts, nranks, dataype, comm, stream)

    switch (comm->device_type) {
        ALL_GATHER_V(INFINI_DEVICE_NVIDIA, cuda);
        ALL_GATHER_V(INFINI_DEVICE_ILUVATAR, cuda);
        ALL_GATHER_V(INFINI_DEVICE_QY, cuda);
        ALL_GATHER_V(INFINI_DEVICE_HYGON, cuda);
        ALL_GATHER_V(INFINI_DEVICE_ASCEND, ascend);
        ALL_GATHER_V(INFINI_DEVICE_CAMBRICON, cambricon);
        ALL_GATHER_V(INFINI_DEVICE_METAX, metax);
        ALL_GATHER_V(INFINI_DEVICE_MOORE, moore);
        ALL_GATHER_V(INFINI_DEVICE_KUNLUN, kunlun);
        ALL_GATHER_V(INFINI_DEVICE_ALI, cuda);

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef ALL_GATHER_V
}

__INFINI_C infiniStatus_t infinicclReduceScatter(
    void *sendbuf,
    void *recvbuf,
    size_t recv_count,
    infiniDtype_t dataype,
    infinicclReduceOp_t op,
    infinicclComm_t comm,
    infinirtStream_t stream) {

    if (comm == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }

#define REDUCE_SCATTER(CASE_, NAMESPACE_) \
    case CASE_:                           \
        return infiniccl::NAMESPACE_::reduceScatter(sendbuf, recvbuf, recv_count, dataype, op, comm, stream)

    switch (comm->device_type) {
        REDUCE_SCATTER(INFINI_DEVICE_NVIDIA, cuda);
        REDUCE_SCATTER(INFINI_DEVICE_ILUVATAR, cuda);
        REDUCE_SCATTER(INFINI_DEVICE_QY, cuda);
        REDUCE_SCATTER(INFINI_DEVICE_HYGON, cuda);
        REDUCE_SCATTER(INFINI_DEVICE_ASCEND, ascend);
        REDUCE_SCATTER(INFINI_DEVICE_CAMBRICON, cambricon);
        REDUCE_SCATTER(INFINI_DEVICE_METAX, metax);
        REDUCE_SCATTER(INFINI_DEVICE_MOORE, moore);
        REDUCE_SCATTER(INFINI_DEVICE_KUNLUN, kunlun);
        REDUCE_SCATTER(INFINI_DEVICE_ALI, cuda);

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef REDUCE_SCATTER
}

__INFINI_C infiniStatus_t infinicclReduceScatterV(
    void *sendbuf,
    void *recvbuf,
    const size_t *send_counts,
    int nranks,
    infiniDtype_t dataype,
    infinicclReduceOp_t op,
    infinicclComm_t comm,
    infinirtStream_t stream) {

    if (comm == nullptr || send_counts == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }

#define REDUCE_SCATTER_V(CASE_, NAMESPACE_) \
    case CASE_:                             \
        return infiniccl::NAMESPACE_::reduceScatterV(sendbuf, recvbuf, send_counts, nranks, dataype, op, comm, stream)

    switch (comm->device_type) {
        REDUCE_SCATTER_V(INFINI_DEVICE_NVIDIA, cuda);
        REDUCE_SCATTER_V(INFINI_DEVICE_ILUVATAR, cuda);
        REDUCE_SCATTER_V(INFINI_DEVICE_QY, cuda);
        REDUCE_SCATTER_V(INFINI_DEVICE_HYGON, cuda);
        REDUCE_SCATTER_V(INFINI_DEVICE_ASCEND, ascend);
        REDUCE_SCATTER_V(INFINI_DEVICE_CAMBRICON, cambricon);
        REDUCE_SCATTER_V(INFINI_DEVICE_METAX, metax);
        REDUCE_SCATTER_V(INFINI_DEVICE_MOORE, moore);
        REDUCE_SCATTER_V(INFINI_DEVICE_KUNLUN, kunlun);
        REDUCE_SCATTER_V(INFINI_DEVICE_ALI, cuda);

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef REDUCE_SCATTER_V
}
