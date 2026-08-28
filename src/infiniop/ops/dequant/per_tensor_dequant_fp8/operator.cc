#include "../../../operator.h"
#include "../../../handle.h"
#include "infiniop/ops/dequant/per_tensor_dequant_fp8.h"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_QY_API)
#include "nvidia/per_tensor_dequant_fp8_nvidia.cuh"
#endif

__INFINI_C infiniStatus_t infiniopCreatePerTensorDequantFp8Descriptor(infiniopHandle_t handle,
                                                                      infiniopPerTensorDequantFp8Descriptor_t *desc_ptr,
                                                                      infiniopTensorDescriptor_t x_desc,
                                                                      infiniopTensorDescriptor_t x_packed_desc,
                                                                      infiniopTensorDescriptor_t x_scale_desc) {
#define CREATE(CASE, NAMESPACE)                                                                  \
    case CASE:                                                                                   \
        return op::per_tensor_dequant_fp8::NAMESPACE::Descriptor::create(                        \
            handle,                                                                              \
            reinterpret_cast<op::per_tensor_dequant_fp8::NAMESPACE::Descriptor **>(desc_ptr),    \
            x_desc,                                                                              \
            x_packed_desc,                                                                       \
            x_scale_desc);
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

__INFINI_C infiniStatus_t infiniopGetPerTensorDequantFp8WorkspaceSize(infiniopPerTensorDequantFp8Descriptor_t desc, size_t *size) {
    switch (desc->device_type) {
#define GET(CASE, NAMESPACE)                                                                                         \
    case CASE:                                                                                                       \
        *size = reinterpret_cast<op::per_tensor_dequant_fp8::NAMESPACE::Descriptor *>(desc)->minWorkspaceSize();     \
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

__INFINI_C infiniStatus_t infiniopPerTensorDequantFp8(infiniopPerTensorDequantFp8Descriptor_t desc,
                                                      void *workspace,
                                                      size_t workspace_size,
                                                      void *x,
                                                      const void *x_packed,
                                                      const void *x_scale,
                                                      void *stream) {
#define DEQUANT(CASE, NAMESPACE)                                                                         \
    case CASE:                                                                                           \
        return reinterpret_cast<op::per_tensor_dequant_fp8::NAMESPACE::Descriptor *>(desc)->calculate(   \
            workspace, workspace_size, x, x_packed, x_scale, stream);

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        DEQUANT(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_QY_API
        DEQUANT(INFINI_DEVICE_QY, nvidia)
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef DEQUANT
}

__INFINI_C infiniStatus_t infiniopDestroyPerTensorDequantFp8Descriptor(infiniopPerTensorDequantFp8Descriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                                                \
    case CASE:                                                                                  \
        delete reinterpret_cast<op::per_tensor_dequant_fp8::NAMESPACE::Descriptor *>(desc);     \
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
