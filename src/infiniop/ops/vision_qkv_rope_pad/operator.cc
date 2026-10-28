#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/vision_qkv_rope_pad.h"
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
#include "nvidia/vision_qkv_rope_pad_nvidia.cuh"
#endif
__INFINI_C infiniStatus_t infiniopCreateVisionQKVRoPEPadDescriptor(infiniopHandle_t handle, infiniopVisionQKVRoPEPadDescriptor_t *desc_ptr, infiniopTensorDescriptor_t q_out, infiniopTensorDescriptor_t k_out, infiniopTensorDescriptor_t v_out, infiniopTensorDescriptor_t qkv, infiniopTensorDescriptor_t cos, infiniopTensorDescriptor_t sin, infiniopTensorDescriptor_t rows, infiniopTensorDescriptor_t cols) {
    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
    case INFINI_DEVICE_NVIDIA: {
        return op::vision_qkv_rope_pad::nvidia::Descriptor::create(handle, reinterpret_cast<op::vision_qkv_rope_pad::nvidia::Descriptor **>(desc_ptr), q_out, k_out, v_out, qkv, cos, sin, rows, cols);
    }
#endif
#ifdef ENABLE_ILUVATAR_API
    case INFINI_DEVICE_ILUVATAR: {
        return op::vision_qkv_rope_pad::nvidia::Descriptor::create(handle, reinterpret_cast<op::vision_qkv_rope_pad::nvidia::Descriptor **>(desc_ptr), q_out, k_out, v_out, qkv, cos, sin, rows, cols);
    }
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}
__INFINI_C infiniStatus_t infiniopGetVisionQKVRoPEPadWorkspaceSize(infiniopVisionQKVRoPEPadDescriptor_t desc, size_t *size) {
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
__INFINI_C infiniStatus_t infiniopVisionQKVRoPEPad(infiniopVisionQKVRoPEPadDescriptor_t desc, void *workspace, size_t workspace_size, void *q_out, void *k_out, void *v_out, const void *qkv, const void *cos, const void *sin, const void *rows, const void *cols, void *stream) {
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
    case INFINI_DEVICE_NVIDIA: {
        return reinterpret_cast<const op::vision_qkv_rope_pad::nvidia::Descriptor *>(desc)->calculate(workspace, workspace_size, q_out, k_out, v_out, qkv, cos, sin, rows, cols, stream);
    }
#endif
#ifdef ENABLE_ILUVATAR_API
    case INFINI_DEVICE_ILUVATAR: {
        return reinterpret_cast<const op::vision_qkv_rope_pad::nvidia::Descriptor *>(desc)->calculate(workspace, workspace_size, q_out, k_out, v_out, qkv, cos, sin, rows, cols, stream);
    }
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}
__INFINI_C infiniStatus_t infiniopDestroyVisionQKVRoPEPadDescriptor(infiniopVisionQKVRoPEPadDescriptor_t desc) {
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
    case INFINI_DEVICE_NVIDIA: {
        delete reinterpret_cast<op::vision_qkv_rope_pad::nvidia::Descriptor *>(desc);
        return INFINI_STATUS_SUCCESS;
    }
#endif
#ifdef ENABLE_ILUVATAR_API
    case INFINI_DEVICE_ILUVATAR: {
        delete reinterpret_cast<op::vision_qkv_rope_pad::nvidia::Descriptor *>(desc);
        return INFINI_STATUS_SUCCESS;
    }
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}
