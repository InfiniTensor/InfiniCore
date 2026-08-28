#include "../../../operator.h"
#include "../../../handle.h"
#include "infiniop/ops/quant/per_tensor_quant_fp8.h"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_QY_API)
#include "nvidia/per_tensor_quant_fp8_nvidia.cuh"
#endif

__INFINI_C infiniStatus_t infiniopCreatePerTensorQuantFp8Descriptor(infiniopHandle_t handle,
                                                                    infiniopPerTensorQuantFp8Descriptor_t *desc_ptr,
                                                                    infiniopTensorDescriptor_t x_packed_desc,
                                                                    infiniopTensorDescriptor_t x_scale_desc,
                                                                    infiniopTensorDescriptor_t x_desc) {
#define CREATE(CASE, NAMESPACE)                                                              \
    case CASE:                                                                               \
        return op::per_tensor_quant_fp8::NAMESPACE::Descriptor::create(                      \
            handle,                                                                          \
            reinterpret_cast<op::per_tensor_quant_fp8::NAMESPACE::Descriptor **>(desc_ptr),  \
            x_packed_desc,                                                                   \
            x_scale_desc,                                                                    \
            x_desc);
    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_QY_API
        CREATE(INFINI_DEVICE_QY, nvidia)
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CREATE
}

__INFINI_C infiniStatus_t infiniopGetPerTensorQuantFp8WorkspaceSize(infiniopPerTensorQuantFp8Descriptor_t desc, size_t *size) {
    switch (desc->device_type) {
#define GET(CASE, NAMESPACE)                                                                                     \
    case CASE:                                                                                                   \
        *size = reinterpret_cast<op::per_tensor_quant_fp8::NAMESPACE::Descriptor *>(desc)->minWorkspaceSize();    \
        return INFINI_STATUS_SUCCESS;
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_QY_API
        GET(INFINI_DEVICE_QY, nvidia)
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET
}

__INFINI_C infiniStatus_t infiniopPerTensorQuantFp8(infiniopPerTensorQuantFp8Descriptor_t desc,
                                                    void *workspace,
                                                    size_t workspace_size,
                                                    void *x_packed,
                                                    void *x_scale,
                                                    const void *x,
                                                    const bool is_static,
                                                    void *stream) {
#define QUANT(CASE, NAMESPACE)                                                                         \
    case CASE:                                                                                         \
        return reinterpret_cast<op::per_tensor_quant_fp8::NAMESPACE::Descriptor *>(desc)->calculate(   \
            workspace, workspace_size, x_packed, x_scale, x, is_static, stream);

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        QUANT(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_QY_API
        QUANT(INFINI_DEVICE_QY, nvidia)
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef QUANT
}

__INFINI_C infiniStatus_t infiniopDestroyPerTensorQuantFp8Descriptor(infiniopPerTensorQuantFp8Descriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                                            \
    case CASE:                                                                              \
        delete reinterpret_cast<op::per_tensor_quant_fp8::NAMESPACE::Descriptor *>(desc);   \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_QY_API
        DESTROY(INFINI_DEVICE_QY, nvidia)
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef DESTROY
}
