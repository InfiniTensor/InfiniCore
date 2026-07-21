#include "infiniccl_mars.h"

#include "../../utils.h"

#include <hccl.h>
#include <hcr/hc_runtime_api.h>

#include <cstdlib>
#include <vector>

#define CHECK_HCCL(API__) CHECK_INTERNAL(API__, hcclSuccess)

namespace {

hcStream_t getMarsStream(infinirtStream_t stream) {
    return stream == nullptr ? nullptr : static_cast<hcStream_t>(stream);
}

hcclDataType_t getHcclDtype(infiniDtype_t datatype) {
    switch (datatype) {
    case INFINI_DTYPE_F32:
        return hcclFloat;
    case INFINI_DTYPE_F16:
        return hcclHalf;
    case INFINI_DTYPE_BF16:
        return hcclBfloat16;
    default:
        std::abort();
    }
}

hcclRedOp_t getHcclRedOp(infinicclReduceOp_t op) {
    switch (op) {
    case INFINICCL_SUM:
        return hcclSum;
    case INFINICCL_PROD:
        return hcclProd;
    case INFINICCL_MAX:
        return hcclMax;
    case INFINICCL_MIN:
        return hcclMin;
    case INFINICCL_AVG:
        return hcclAvg;
    default:
        std::abort();
    }
}

hcclComm_t getHcclComm(infinicclComm_t comm) {
    return static_cast<hcclComm_t>(comm->comm);
}

} // namespace

namespace infiniccl::mars {

infiniStatus_t commInitAll(infinicclComm_t *comms, int ndevice, const int *device_ids) {
    std::vector<hcclComm_t> hccl_comms(ndevice);
    CHECK_HCCL(hcclCommInitAll(hccl_comms.data(), ndevice, device_ids));

    for (int i = 0; i < ndevice; ++i) {
        comms[i] = new InfinicclComm{INFINI_DEVICE_MARS, device_ids[i], hccl_comms[i]};
    }
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t commDestroy(infinicclComm_t comm) {
    CHECK_HCCL(hcclCommDestroy(getHcclComm(comm)));
    delete comm;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t allReduce(void *sendbuf,
                         void *recvbuf,
                         size_t count,
                         infiniDtype_t datatype,
                         infinicclReduceOp_t op,
                         infinicclComm_t comm,
                         infinirtStream_t stream) {
    CHECK_DTYPE(datatype, INFINI_DTYPE_F32, INFINI_DTYPE_F16, INFINI_DTYPE_BF16);
    CHECK_HCCL(hcclAllReduce(sendbuf,
                             recvbuf,
                             count,
                             getHcclDtype(datatype),
                             getHcclRedOp(op),
                             getHcclComm(comm),
                             getMarsStream(stream)));
    return INFINI_STATUS_SUCCESS;
}

INFINICCL_DEVICE_API_UNSUPPORTED_EP_COLLECTIVES();

} // namespace infiniccl::mars
