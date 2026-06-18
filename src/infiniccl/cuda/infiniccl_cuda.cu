#include "infiniccl_cuda.h"

#include <cuda_runtime.h>
#include <iostream>
#include <nccl.h>
#include <vector>

#include "../../utils.h"

#define CHECK_NCCL(API__) CHECK_INTERNAL(API__, ncclSuccess)

inline cudaStream_t getCudaStream(infinirtStream_t stream) {
    if (stream == nullptr) {
        return 0;
    }
    return static_cast<cudaStream_t>(stream);
}

inline ncclDataType_t getNcclDtype(infiniDtype_t datatype) {
    switch (datatype) {
    case INFINI_DTYPE_I32:
        return ncclInt32;
    case INFINI_DTYPE_I64:
        return ncclInt64;
    case INFINI_DTYPE_U32:
        return ncclUint32;
    case INFINI_DTYPE_U64:
        return ncclUint64;
    case INFINI_DTYPE_F32:
        return ncclFloat;
    case INFINI_DTYPE_F16:
        return ncclHalf;
    case INFINI_DTYPE_BF16:
        return ncclBfloat16;
    default:
        std::abort();
        return ncclHalf;
    }
}

inline ncclRedOp_t getNcclRedOp(infinicclReduceOp_t op) {
    switch (op) {
    case INFINICCL_SUM:
        return ncclSum;
    case INFINICCL_PROD:
        return ncclProd;
    case INFINICCL_MAX:
        return ncclMax;
    case INFINICCL_MIN:
        return ncclMin;
    case INFINICCL_AVG:
        return ncclAvg;
    default:
        std::abort();
        return ncclSum;
    }
}

inline ncclComm_t getNcclComm(infinicclComm_t comm) {
    return static_cast<ncclComm_t>(comm->comm);
}

namespace infiniccl::cuda {

infiniStatus_t commInitAll(
    infinicclComm_t *comms,
    int ndevice,
    const int *device_ids) {

    std::vector<ncclComm_t> nccl_comms(ndevice);
    CHECK_NCCL(ncclCommInitAll(nccl_comms.data(), ndevice, (int const *)device_ids));

    for (int i = 0; i < ndevice; i++) {
        comms[i] = new InfinicclComm{INFINI_DEVICE_NVIDIA, device_ids[i], (void *)(nccl_comms[i]), i, ndevice};
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t commDestroy(infinicclComm_t comm) {
    CHECK_NCCL(ncclCommDestroy(getNcclComm(comm)));
    delete comm;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t groupStart(infinicclComm_t) {
    CHECK_NCCL(ncclGroupStart());
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t groupEnd(infinicclComm_t) {
    CHECK_NCCL(ncclGroupEnd());
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t allReduce(
    void *sendbuf,
    void *recvbuf,
    size_t count,
    infiniDtype_t datatype,
    infinicclReduceOp_t op,
    infinicclComm_t comm,
    infinirtStream_t stream) {

    CHECK_DTYPE(datatype, INFINI_DTYPE_F32, INFINI_DTYPE_F16, INFINI_DTYPE_BF16);

    CHECK_NCCL(ncclAllReduce(sendbuf, recvbuf, count, getNcclDtype(datatype),
                             getNcclRedOp(op), getNcclComm(comm), getCudaStream(stream)));

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t allGather(
    void *sendbuf,
    void *recvbuf,
    size_t send_count,
    infiniDtype_t datatype,
    infinicclComm_t comm,
    infinirtStream_t stream) {

    CHECK_DTYPE(datatype, INFINI_DTYPE_F32, INFINI_DTYPE_F16, INFINI_DTYPE_BF16,
                INFINI_DTYPE_I32, INFINI_DTYPE_I64, INFINI_DTYPE_U32, INFINI_DTYPE_U64);

    CHECK_NCCL(ncclAllGather(sendbuf, recvbuf, send_count, getNcclDtype(datatype),
                             getNcclComm(comm), getCudaStream(stream)));

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t allGatherV(
    void *sendbuf,
    void *recvbuf,
    const size_t *recv_counts,
    int nranks,
    infiniDtype_t datatype,
    infinicclComm_t comm,
    infinirtStream_t stream) {

    CHECK_DTYPE(datatype, INFINI_DTYPE_F32, INFINI_DTYPE_F16, INFINI_DTYPE_BF16,
                INFINI_DTYPE_I32, INFINI_DTYPE_I64, INFINI_DTYPE_U32, INFINI_DTYPE_U64);
    CHECK_OR_DO(nranks == comm->world_size, return INFINI_STATUS_BAD_PARAM);

    auto cuda_stream = getCudaStream(stream);
    ncclComm_t nccl_comm = getNcclComm(comm);
    ncclDataType_t nccl_dtype = getNcclDtype(datatype);
    size_t offset = 0;

    CHECK_NCCL(ncclGroupStart());
    for (int root = 0; root < nranks; ++root) {
        CHECK_NCCL(ncclBroadcast(
            sendbuf,
            static_cast<char *>(recvbuf) + offset,
            recv_counts[root],
            nccl_dtype,
            root,
            nccl_comm,
            cuda_stream));
        offset += recv_counts[root] * infiniSizeOf(datatype);
    }
    CHECK_NCCL(ncclGroupEnd());

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t reduceScatter(
    void *sendbuf,
    void *recvbuf,
    size_t recv_count,
    infiniDtype_t datatype,
    infinicclReduceOp_t op,
    infinicclComm_t comm,
    infinirtStream_t stream) {

    CHECK_DTYPE(datatype, INFINI_DTYPE_F32, INFINI_DTYPE_F16, INFINI_DTYPE_BF16,
                INFINI_DTYPE_I32, INFINI_DTYPE_I64, INFINI_DTYPE_U32, INFINI_DTYPE_U64);

    CHECK_NCCL(ncclReduceScatter(sendbuf, recvbuf, recv_count, getNcclDtype(datatype),
                                 getNcclRedOp(op), getNcclComm(comm), getCudaStream(stream)));

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t reduceScatterV(
    void *sendbuf,
    void *recvbuf,
    const size_t *send_counts,
    int nranks,
    infiniDtype_t datatype,
    infinicclReduceOp_t op,
    infinicclComm_t comm,
    infinirtStream_t stream) {

    CHECK_DTYPE(datatype, INFINI_DTYPE_F32, INFINI_DTYPE_F16, INFINI_DTYPE_BF16,
                INFINI_DTYPE_I32, INFINI_DTYPE_I64, INFINI_DTYPE_U32, INFINI_DTYPE_U64);
    CHECK_OR_DO(nranks == comm->world_size, return INFINI_STATUS_BAD_PARAM);

    auto cuda_stream = getCudaStream(stream);
    ncclComm_t nccl_comm = getNcclComm(comm);
    ncclDataType_t nccl_dtype = getNcclDtype(datatype);
    ncclRedOp_t nccl_op = getNcclRedOp(op);
    size_t offset = 0;

    CHECK_NCCL(ncclGroupStart());
    for (int root = 0; root < nranks; ++root) {
        CHECK_NCCL(ncclReduce(
            static_cast<char *>(sendbuf) + offset,
            recvbuf,
            send_counts[root],
            nccl_dtype,
            nccl_op,
            root,
            nccl_comm,
            cuda_stream));
        offset += send_counts[root] * infiniSizeOf(datatype);
    }
    CHECK_NCCL(ncclGroupEnd());

    return INFINI_STATUS_SUCCESS;
}
} // namespace infiniccl::cuda
