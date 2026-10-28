#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/qk_rms_norm_rope.h"
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
#include "nvidia/qk_rms_norm_rope_nvidia.cuh"
#endif
__INFINI_C infiniStatus_t infiniopCreateQKRMSNormRoPEDescriptor(infiniopHandle_t handle, infiniopQKRMSNormRoPEDescriptor_t *desc_ptr, infiniopTensorDescriptor_t q_out,
                                                                infiniopTensorDescriptor_t k_out,
                                                                infiniopTensorDescriptor_t q,
                                                                infiniopTensorDescriptor_t k,
                                                                infiniopTensorDescriptor_t q_weight,
                                                                infiniopTensorDescriptor_t k_weight,
                                                                infiniopTensorDescriptor_t cos,
                                                                infiniopTensorDescriptor_t sin,
                                                                infiniopTensorDescriptor_t positions, float epsilon) {
    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
    case INFINI_DEVICE_NVIDIA:
        return op::qk_rms_norm_rope::nvidia::Descriptor::create(handle, reinterpret_cast<op::qk_rms_norm_rope::nvidia::Descriptor **>(desc_ptr), q_out, k_out, q, k, q_weight, k_weight, cos, sin, positions, epsilon);
#endif
#ifdef ENABLE_ILUVATAR_API
    case INFINI_DEVICE_ILUVATAR:
        return op::qk_rms_norm_rope::nvidia::Descriptor::create(handle, reinterpret_cast<op::qk_rms_norm_rope::nvidia::Descriptor **>(desc_ptr), q_out, k_out, q, k, q_weight, k_weight, cos, sin, positions, epsilon);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}
__INFINI_C infiniStatus_t infiniopGetQKRMSNormRoPEWorkspaceSize(infiniopQKRMSNormRoPEDescriptor_t desc, size_t *size) {
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
__INFINI_C infiniStatus_t infiniopQKRMSNormRoPE(infiniopQKRMSNormRoPEDescriptor_t desc, void *workspace, size_t workspace_size, void *q_out,
                                                void *k_out,
                                                const void *q,
                                                const void *k,
                                                const void *q_weight,
                                                const void *k_weight,
                                                const void *cos,
                                                const void *sin,
                                                const void *positions, void *stream) {
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
    case INFINI_DEVICE_NVIDIA:
        return reinterpret_cast<const op::qk_rms_norm_rope::nvidia::Descriptor *>(desc)->calculate(workspace, workspace_size, q_out, k_out, q, k, q_weight, k_weight, cos, sin, positions, stream);
#endif
#ifdef ENABLE_ILUVATAR_API
    case INFINI_DEVICE_ILUVATAR:
        return reinterpret_cast<const op::qk_rms_norm_rope::nvidia::Descriptor *>(desc)->calculate(workspace, workspace_size, q_out, k_out, q, k, q_weight, k_weight, cos, sin, positions, stream);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}
__INFINI_C infiniStatus_t infiniopDestroyQKRMSNormRoPEDescriptor(infiniopQKRMSNormRoPEDescriptor_t desc) {
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
    case INFINI_DEVICE_NVIDIA:
        delete reinterpret_cast<op::qk_rms_norm_rope::nvidia::Descriptor *>(desc);
        return INFINI_STATUS_SUCCESS;
#endif
#ifdef ENABLE_ILUVATAR_API
    case INFINI_DEVICE_ILUVATAR:
        delete reinterpret_cast<op::qk_rms_norm_rope::nvidia::Descriptor *>(desc);
        return INFINI_STATUS_SUCCESS;
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}
