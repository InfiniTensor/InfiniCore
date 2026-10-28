#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/rms_norm_gated.h"
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
#include "nvidia/rms_norm_gated_nvidia.cuh"
#endif
__INFINI_C infiniStatus_t infiniopCreateRMSNormGatedDescriptor(infiniopHandle_t handle, infiniopRMSNormGatedDescriptor_t *desc_ptr, infiniopTensorDescriptor_t out, infiniopTensorDescriptor_t x, infiniopTensorDescriptor_t gate, infiniopTensorDescriptor_t weight, float epsilon) {
    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
    case INFINI_DEVICE_NVIDIA:
        return op::rms_norm_gated::nvidia::Descriptor::create(handle, reinterpret_cast<op::rms_norm_gated::nvidia::Descriptor **>(desc_ptr), out, x, gate, weight, epsilon);
#endif
#ifdef ENABLE_ILUVATAR_API
    case INFINI_DEVICE_ILUVATAR:
        return op::rms_norm_gated::nvidia::Descriptor::create(handle, reinterpret_cast<op::rms_norm_gated::nvidia::Descriptor **>(desc_ptr), out, x, gate, weight, epsilon);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}
__INFINI_C infiniStatus_t infiniopGetRMSNormGatedWorkspaceSize(infiniopRMSNormGatedDescriptor_t desc, size_t *size) {
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
    case INFINI_DEVICE_NVIDIA:
        *size = 0;
        return INFINI_STATUS_SUCCESS;
#endif
#ifdef ENABLE_ILUVATAR_API
    case INFINI_DEVICE_ILUVATAR:
        *size = 0;
        return INFINI_STATUS_SUCCESS;
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}
__INFINI_C infiniStatus_t infiniopRMSNormGated(infiniopRMSNormGatedDescriptor_t desc, void *workspace, size_t workspace_size, void *out, const void *x, const void *gate, const void *weight, void *stream) {
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
    case INFINI_DEVICE_NVIDIA:
        return reinterpret_cast<const op::rms_norm_gated::nvidia::Descriptor *>(desc)->calculate(workspace, workspace_size, out, x, gate, weight, stream);
#endif
#ifdef ENABLE_ILUVATAR_API
    case INFINI_DEVICE_ILUVATAR:
        return reinterpret_cast<const op::rms_norm_gated::nvidia::Descriptor *>(desc)->calculate(workspace, workspace_size, out, x, gate, weight, stream);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}
__INFINI_C infiniStatus_t infiniopDestroyRMSNormGatedDescriptor(infiniopRMSNormGatedDescriptor_t desc) {
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
    case INFINI_DEVICE_NVIDIA:
        delete reinterpret_cast<op::rms_norm_gated::nvidia::Descriptor *>(desc);
        return INFINI_STATUS_SUCCESS;
#endif
#ifdef ENABLE_ILUVATAR_API
    case INFINI_DEVICE_ILUVATAR:
        delete reinterpret_cast<op::rms_norm_gated::nvidia::Descriptor *>(desc);
        return INFINI_STATUS_SUCCESS;
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}
