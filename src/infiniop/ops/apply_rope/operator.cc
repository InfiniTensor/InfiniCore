#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/apply_rope.h"
#ifdef ENABLE_ASCEND_API
#include "ascend/apply_rope_ascend.h"
#endif

__INFINI_C infiniStatus_t infiniopCreateApplyRoPEDescriptor(
    infiniopHandle_t handle, infiniopApplyRoPEDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t q, infiniopTensorDescriptor_t k,
    infiniopTensorDescriptor_t cos, infiniopTensorDescriptor_t sin) {
    if (!handle || !desc_ptr || !q || !k || !cos || !sin) {
        return INFINI_STATUS_NULL_POINTER;
    }
#ifdef ENABLE_ASCEND_API
    if (handle->device == INFINI_DEVICE_ASCEND) {
        return op::apply_rope::ascend::Descriptor::create(
            handle, reinterpret_cast<op::apply_rope::ascend::Descriptor **>(desc_ptr),
            q, k, cos, sin);
    }
#endif
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}
__INFINI_C infiniStatus_t infiniopGetApplyRoPEWorkspaceSize(
    infiniopApplyRoPEDescriptor_t desc, size_t *size) {
    if (!desc || !size) {
        return INFINI_STATUS_NULL_POINTER;
    }
#ifdef ENABLE_ASCEND_API
    if (desc->device_type == INFINI_DEVICE_ASCEND) {
        *size = reinterpret_cast<op::apply_rope::ascend::Descriptor *>(desc)->workspaceSize();
        return INFINI_STATUS_SUCCESS;
    }
#endif
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}
__INFINI_C infiniStatus_t infiniopApplyRoPE(
    infiniopApplyRoPEDescriptor_t desc, void *workspace, size_t workspace_size,
    void *q, void *k, const void *cos, const void *sin, void *stream) {
    if (!desc || !q || !k || !cos || !sin) {
        return INFINI_STATUS_NULL_POINTER;
    }
#ifdef ENABLE_ASCEND_API
    if (desc->device_type == INFINI_DEVICE_ASCEND) {
        return reinterpret_cast<op::apply_rope::ascend::Descriptor *>(desc)->calculate(
            workspace, workspace_size, q, k, cos, sin, stream);
    }
#endif
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}
__INFINI_C infiniStatus_t infiniopDestroyApplyRoPEDescriptor(
    infiniopApplyRoPEDescriptor_t desc) {
    if (!desc) {
        return INFINI_STATUS_NULL_POINTER;
    }
#ifdef ENABLE_ASCEND_API
    if (desc->device_type == INFINI_DEVICE_ASCEND) {
        delete reinterpret_cast<op::apply_rope::ascend::Descriptor *>(desc);
        return INFINI_STATUS_SUCCESS;
    }
#endif
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}
