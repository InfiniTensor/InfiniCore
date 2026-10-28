#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/qk_rms_norm_mrope.h"
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
#include "nvidia/qk_rms_norm_mrope_nvidia.cuh"
#endif
__INFINI_C infiniStatus_t infiniopCreateQKRMSNormMRoPEDescriptor(infiniopHandle_t handle, infiniopQKRMSNormMRoPEDescriptor_t *desc_ptr, infiniopTensorDescriptor_t q_out,
                                                                 infiniopTensorDescriptor_t k_out,
                                                                 infiniopTensorDescriptor_t q,
                                                                 infiniopTensorDescriptor_t k,
                                                                 infiniopTensorDescriptor_t q_weight,
                                                                 infiniopTensorDescriptor_t k_weight,
                                                                 infiniopTensorDescriptor_t cos,
                                                                 infiniopTensorDescriptor_t sin,
                                                                 infiniopTensorDescriptor_t positions, float epsilon, int section_t, int section_h, int section_w, bool interleaved) {
    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
    case INFINI_DEVICE_NVIDIA:
        return op::qk_rms_norm_mrope::nvidia::Descriptor::create(handle, reinterpret_cast<op::qk_rms_norm_mrope::nvidia::Descriptor **>(desc_ptr), q_out, k_out, q, k, q_weight, k_weight, cos, sin, positions, epsilon, section_t, section_h, section_w, interleaved);
#endif
#ifdef ENABLE_ILUVATAR_API
    case INFINI_DEVICE_ILUVATAR:
        return op::qk_rms_norm_mrope::nvidia::Descriptor::create(handle, reinterpret_cast<op::qk_rms_norm_mrope::nvidia::Descriptor **>(desc_ptr), q_out, k_out, q, k, q_weight, k_weight, cos, sin, positions, epsilon, section_t, section_h, section_w, interleaved);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}
__INFINI_C infiniStatus_t infiniopGetQKRMSNormMRoPEWorkspaceSize(infiniopQKRMSNormMRoPEDescriptor_t desc, size_t *size) {
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
__INFINI_C infiniStatus_t infiniopQKRMSNormMRoPE(infiniopQKRMSNormMRoPEDescriptor_t desc, void *workspace, size_t workspace_size, void *q_out,
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
        return reinterpret_cast<const op::qk_rms_norm_mrope::nvidia::Descriptor *>(desc)->calculate(workspace, workspace_size, q_out, k_out, q, k, q_weight, k_weight, cos, sin, positions, stream);
#endif
#ifdef ENABLE_ILUVATAR_API
    case INFINI_DEVICE_ILUVATAR:
        return reinterpret_cast<const op::qk_rms_norm_mrope::nvidia::Descriptor *>(desc)->calculate(workspace, workspace_size, q_out, k_out, q, k, q_weight, k_weight, cos, sin, positions, stream);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}
__INFINI_C infiniStatus_t infiniopDestroyQKRMSNormMRoPEDescriptor(infiniopQKRMSNormMRoPEDescriptor_t desc) {
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
    case INFINI_DEVICE_NVIDIA:
        delete reinterpret_cast<op::qk_rms_norm_mrope::nvidia::Descriptor *>(desc);
        return INFINI_STATUS_SUCCESS;
#endif
#ifdef ENABLE_ILUVATAR_API
    case INFINI_DEVICE_ILUVATAR:
        delete reinterpret_cast<op::qk_rms_norm_mrope::nvidia::Descriptor *>(desc);
        return INFINI_STATUS_SUCCESS;
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}
