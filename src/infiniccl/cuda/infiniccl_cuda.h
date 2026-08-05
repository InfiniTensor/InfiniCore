#ifndef INFINICCL_CUDA_H_
#define INFINICCL_CUDA_H_

#include "../infiniccl_impl.h"

// Windows does not support CUDA
#if (defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API) || defined(ENABLE_ALI_API)) && defined(ENABLE_CCL) && !defined(_WIN32)
INFINICCL_DEVICE_API_IMPL(cuda)
namespace infiniccl::cuda {
infiniStatus_t getUniqueId(infinicclUniqueId_t *unique_id);
infiniStatus_t commInitRank(
    infinicclComm_t *comm,
    int nranks,
    infinicclUniqueId_t comm_id,
    int rank);
infiniStatus_t broadcast(
    const void *sendbuf,
    void *recvbuf,
    size_t count,
    infiniDtype_t datatype,
    int root,
    infinicclComm_t comm,
    infinirtStream_t stream);
infiniStatus_t send(
    const void *sendbuf,
    size_t count,
    infiniDtype_t datatype,
    int peer,
    infinicclComm_t comm,
    infinirtStream_t stream);
infiniStatus_t recv(
    void *recvbuf,
    size_t count,
    infiniDtype_t datatype,
    int peer,
    infinicclComm_t comm,
    infinirtStream_t stream);
} // namespace infiniccl::cuda
#else
INFINICCL_DEVICE_API_NOOP(cuda)
namespace infiniccl::cuda {
inline infiniStatus_t getUniqueId(infinicclUniqueId_t *) {
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}
inline infiniStatus_t commInitRank(
    infinicclComm_t *,
    int,
    infinicclUniqueId_t,
    int) {
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}
inline infiniStatus_t broadcast(
    const void *,
    void *,
    size_t,
    infiniDtype_t,
    int,
    infinicclComm_t,
    infinirtStream_t) {
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}
inline infiniStatus_t send(
    const void *,
    size_t,
    infiniDtype_t,
    int,
    infinicclComm_t,
    infinirtStream_t) {
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}
inline infiniStatus_t recv(
    void *,
    size_t,
    infiniDtype_t,
    int,
    infinicclComm_t,
    infinirtStream_t) {
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}
} // namespace infiniccl::cuda
#endif

#endif /* INFINICCL_CUDA_H_ */
