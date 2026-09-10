#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/linear_gguf.h"

#if defined ENABLE_NVIDIA_API
#include "nvidia/linear_gguf_nvidia.cuh"
#endif

__INFINI_C infiniStatus_t infiniopCreateLinearGgufDescriptor(
    infiniopHandle_t handle,
    infiniopLinearGgufDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t w_desc,
    int64_t ggml_type) {
#define CREATE(CASE, NAMESPACE)                                                      \
    case CASE:                                                                       \
        return op::linear_gguf::NAMESPACE::Descriptor::create(                       \
            handle,                                                                  \
            reinterpret_cast<op::linear_gguf::NAMESPACE::Descriptor **>(desc_ptr),    \
            out_desc,                                                                \
            a_desc,                                                                  \
            w_desc,                                                                  \
            ggml_type)

    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__INFINI_C infiniStatus_t infiniopGetLinearGgufWorkspaceSize(infiniopLinearGgufDescriptor_t desc,
                                                            size_t *size) {
#define GET(CASE, NAMESPACE)                                                                                \
    case CASE:                                                                                              \
        *size = reinterpret_cast<const op::linear_gguf::NAMESPACE::Descriptor *>(desc)->workspaceSize();    \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET
}

__INFINI_C infiniStatus_t infiniopLinearGguf(
    infiniopLinearGgufDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *a,
    const void *w,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                        \
    case CASE:                                                                            \
        return reinterpret_cast<const op::linear_gguf::NAMESPACE::Descriptor *>(desc)     \
            ->calculate(workspace, workspace_size, out, a, w, stream)

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__INFINI_C infiniStatus_t
infiniopDestroyLinearGgufDescriptor(infiniopLinearGgufDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                            \
    case CASE:                                                                             \
        delete reinterpret_cast<op::linear_gguf::NAMESPACE::Descriptor *>(desc);                 \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        DELETE(INFINI_DEVICE_NVIDIA, nvidia);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DELETE
}
