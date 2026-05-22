#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/simple_gla_prefill.h"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/simple_gla_prefill_nvidia_cuda.cuh"
#endif

__INFINI_C infiniStatus_t infiniopCreateSimpleGLAPrefillDescriptor(
    infiniopHandle_t handle,
    infiniopSimpleGLAPrefillDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_desc,
    infiniopTensorDescriptor_t v_desc,
    infiniopTensorDescriptor_t g_gamma_desc) {

    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
    case INFINI_DEVICE_NVIDIA:
        return op::simple_gla_prefill_cuda::nvidia::Descriptor::create(
            handle,
            reinterpret_cast<op::simple_gla_prefill_cuda::nvidia::Descriptor **>(desc_ptr),
            out_desc, q_desc, k_desc, v_desc, g_gamma_desc);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}

__INFINI_C infiniStatus_t infiniopGetSimpleGLAPrefillWorkspaceSize(
    infiniopSimpleGLAPrefillDescriptor_t desc,
    size_t *size) {

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
    case INFINI_DEVICE_NVIDIA:
        *size = reinterpret_cast<op::simple_gla_prefill_cuda::nvidia::Descriptor *>(desc)->workspaceSize();
        return INFINI_STATUS_SUCCESS;
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}

__INFINI_C infiniStatus_t infiniopSimpleGLAPrefill(
    infiniopSimpleGLAPrefillDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    void const *q,
    void const *k,
    void const *v,
    void const *g_gamma,
    float scale,
    void *stream) {

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
    case INFINI_DEVICE_NVIDIA:
        return reinterpret_cast<const op::simple_gla_prefill_cuda::nvidia::Descriptor *>(desc)
            ->calculate(workspace, workspace_size, out, q, k, v, g_gamma, scale, stream);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}

__INFINI_C infiniStatus_t infiniopDestroySimpleGLAPrefillDescriptor(
    infiniopSimpleGLAPrefillDescriptor_t desc) {

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
    case INFINI_DEVICE_NVIDIA:
        delete reinterpret_cast<const op::simple_gla_prefill_cuda::nvidia::Descriptor *>(desc);
        return INFINI_STATUS_SUCCESS;
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}
