#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/mxfp4_dequantize.h"

#ifdef ENABLE_CPU_API
#include "cpu/mxfp4_dequantize_cpu.h"
#endif
#ifdef ENABLE_NVIDIA_API
#include "nvidia/mxfp4_dequantize_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/mxfp4_dequantize_metax.h"
#endif
#ifdef ENABLE_MOORE_API
#include "moore/mxfp4_dequantize_moore.h"
#endif

__INFINI_C infiniStatus_t infiniopCreateMxfp4DequantizeDescriptor(
    infiniopHandle_t handle,
    infiniopMxfp4DequantizeDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t packed_desc,
    infiniopTensorDescriptor_t scales_desc) {
#define CREATE(CASE, NAMESPACE)                                                         \
    case CASE:                                                                          \
        return op::mxfp4_dequantize::NAMESPACE::Descriptor::create(                     \
            handle,                                                                     \
            reinterpret_cast<op::mxfp4_dequantize::NAMESPACE::Descriptor **>(desc_ptr), \
            out_desc, packed_desc, scales_desc)
    switch (handle->device) {
#ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_METAX_API
        CREATE(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_MOORE_API
        CREATE(INFINI_DEVICE_MOORE, moore);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CREATE
}

__INFINI_C infiniStatus_t infiniopGetMxfp4DequantizeWorkspaceSize(
    infiniopMxfp4DequantizeDescriptor_t desc,
    size_t *size) {
#define GET(CASE, NAMESPACE)                                                                \
    case CASE:                                                                              \
        *size = reinterpret_cast<const op::mxfp4_dequantize::NAMESPACE::Descriptor *>(desc) \
                    ->workspaceSize();                                                      \
        return INFINI_STATUS_SUCCESS
    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        GET(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_METAX_API
        GET(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_MOORE_API
        GET(INFINI_DEVICE_MOORE, moore);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET
}

__INFINI_C infiniStatus_t infiniopMxfp4Dequantize(
    infiniopMxfp4DequantizeDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *packed,
    const void *scales,
    void *stream) {
#define CALCULATE(CASE, NAMESPACE)                                                         \
    case CASE:                                                                             \
        return reinterpret_cast<const op::mxfp4_dequantize::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, out, packed, scales, stream)
    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_METAX_API
        CALCULATE(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_MOORE_API
        CALCULATE(INFINI_DEVICE_MOORE, moore);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CALCULATE
}

__INFINI_C infiniStatus_t infiniopDestroyMxfp4DequantizeDescriptor(
    infiniopMxfp4DequantizeDescriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                                            \
    case CASE:                                                                              \
        delete reinterpret_cast<const op::mxfp4_dequantize::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS
    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        DESTROY(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_METAX_API
        DESTROY(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_MOORE_API
        DESTROY(INFINI_DEVICE_MOORE, moore);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef DESTROY
}
