#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/gdn_qkv_norm.h"
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
#include "nvidia/gdn_qkv_norm_nvidia.cuh"
#endif
__INFINI_C infiniStatus_t infiniopCreateGDNQKVNormDescriptor(infiniopHandle_t handle, infiniopGDNQKVNormDescriptor_t *desc_ptr, infiniopTensorDescriptor_t q_out, infiniopTensorDescriptor_t k_out, infiniopTensorDescriptor_t v_out, infiniopTensorDescriptor_t q, infiniopTensorDescriptor_t k, infiniopTensorDescriptor_t v) {
    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
    case INFINI_DEVICE_NVIDIA: {
        return op::gdn_qkv_norm::nvidia::Descriptor::create(handle, reinterpret_cast<op::gdn_qkv_norm::nvidia::Descriptor **>(desc_ptr), q_out, k_out, v_out, q, k, v);
    }
#endif
#ifdef ENABLE_ILUVATAR_API
    case INFINI_DEVICE_ILUVATAR: {
        return op::gdn_qkv_norm::nvidia::Descriptor::create(handle, reinterpret_cast<op::gdn_qkv_norm::nvidia::Descriptor **>(desc_ptr), q_out, k_out, v_out, q, k, v);
    }
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}
__INFINI_C infiniStatus_t infiniopGetGDNQKVNormWorkspaceSize(infiniopGDNQKVNormDescriptor_t desc, size_t *size) {
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
    case INFINI_DEVICE_NVIDIA: {
        *size = 0;
        return INFINI_STATUS_SUCCESS;
    }
#endif
#ifdef ENABLE_ILUVATAR_API
    case INFINI_DEVICE_ILUVATAR: {
        *size = 0;
        return INFINI_STATUS_SUCCESS;
    }
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}
__INFINI_C infiniStatus_t infiniopGDNQKVNorm(infiniopGDNQKVNormDescriptor_t desc, void *workspace, size_t workspace_size, void *q_out, void *k_out, void *v_out, const void *q, const void *k, const void *v, void *stream) {
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
    case INFINI_DEVICE_NVIDIA: {
        return reinterpret_cast<const op::gdn_qkv_norm::nvidia::Descriptor *>(desc)->calculate(workspace, workspace_size, q_out, k_out, v_out, q, k, v, stream);
    }
#endif
#ifdef ENABLE_ILUVATAR_API
    case INFINI_DEVICE_ILUVATAR: {
        return reinterpret_cast<const op::gdn_qkv_norm::nvidia::Descriptor *>(desc)->calculate(workspace, workspace_size, q_out, k_out, v_out, q, k, v, stream);
    }
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}
__INFINI_C infiniStatus_t infiniopDestroyGDNQKVNormDescriptor(infiniopGDNQKVNormDescriptor_t desc) {
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
    case INFINI_DEVICE_NVIDIA: {
        delete reinterpret_cast<op::gdn_qkv_norm::nvidia::Descriptor *>(desc);
        return INFINI_STATUS_SUCCESS;
    }
#endif
#ifdef ENABLE_ILUVATAR_API
    case INFINI_DEVICE_ILUVATAR: {
        delete reinterpret_cast<op::gdn_qkv_norm::nvidia::Descriptor *>(desc);
        return INFINI_STATUS_SUCCESS;
    }
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}
