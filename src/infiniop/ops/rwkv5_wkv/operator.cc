#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/rwkv5_wkv.h"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_ALI_API) || defined(ENABLE_HYGON_API)
#include "nvidia/rwkv5_wkv_nvidia.cuh"
#endif

__INFINI_C infiniStatus_t infiniopCreateRwkv5WkvDescriptor(
    infiniopHandle_t handle,
    infiniopRwkv5WkvDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t receptance_desc,
    infiniopTensorDescriptor_t key_desc,
    infiniopTensorDescriptor_t value_desc,
    infiniopTensorDescriptor_t time_decay_desc,
    infiniopTensorDescriptor_t time_faaaa_desc,
    infiniopTensorDescriptor_t state_desc) {

#define CREATE(CASE, NAMESPACE)                                                          \
    case CASE:                                                                           \
        return op::rwkv5_wkv::NAMESPACE::Descriptor::create(                             \
            handle, reinterpret_cast<op::rwkv5_wkv::NAMESPACE::Descriptor **>(desc_ptr), \
            out_desc, receptance_desc, key_desc, value_desc, time_decay_desc,            \
            time_faaaa_desc, state_desc)

    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CREATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_QY_API
        CREATE(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_ALI_API
        CREATE(INFINI_DEVICE_ALI, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        CREATE(INFINI_DEVICE_HYGON, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CREATE
}

__INFINI_C infiniStatus_t infiniopGetRwkv5WkvWorkspaceSize(
    infiniopRwkv5WkvDescriptor_t desc,
    size_t *size) {

#define GET(CASE, NAMESPACE)                                                                     \
    case CASE:                                                                                   \
        *size = reinterpret_cast<op::rwkv5_wkv::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        GET(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_QY_API
        GET(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_ALI_API
        GET(INFINI_DEVICE_ALI, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        GET(INFINI_DEVICE_HYGON, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET
}

__INFINI_C infiniStatus_t infiniopRwkv5Wkv(
    infiniopRwkv5WkvDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *receptance,
    const void *key,
    const void *value,
    const void *time_decay,
    const void *time_faaaa,
    void *state,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                          \
    case CASE:                                                                              \
        return reinterpret_cast<op::rwkv5_wkv::NAMESPACE::Descriptor *>(desc)->calculate(   \
            workspace, workspace_size, out, receptance, key, value, time_decay, time_faaaa, \
            state, stream)

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CALCULATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_QY_API
        CALCULATE(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_ALI_API
        CALCULATE(INFINI_DEVICE_ALI, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        CALCULATE(INFINI_DEVICE_HYGON, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CALCULATE
}

__INFINI_C infiniStatus_t infiniopDestroyRwkv5WkvDescriptor(
    infiniopRwkv5WkvDescriptor_t desc) {

#define DESTROY(CASE, NAMESPACE)                                               \
    case CASE:                                                                 \
        delete reinterpret_cast<op::rwkv5_wkv::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        DESTROY(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_QY_API
        DESTROY(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_ALI_API
        DESTROY(INFINI_DEVICE_ALI, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        DESTROY(INFINI_DEVICE_HYGON, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef DESTROY
}
