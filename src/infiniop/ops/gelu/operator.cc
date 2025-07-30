#include "../../handle.h"
#include "infinicore.h"
#include "infiniop/ops/gelu.h"

#ifdef ENABLE_CPU_API
#include "cpu/gelu_cpu.h"
#endif
#ifdef ENABLE_ILUVATAR_API
#include "nvidia/gelu_nvidia.cuh"
#endif
#ifdef ENABLE_NVIDIA_API
#include "nvidia/gelu_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/gelu_metax.h"
#endif

__C infiniStatus_t infiniopCreateGeluDescriptor(
    infiniopHandle_t handle,
    infiniopGeluDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc) {
#define CTEATE(CASE, NAMESPACE)                                             \
    case CASE:                                                              \
        return op::gelu::NAMESPACE::Descriptor::create(                     \
            handle,                                                         \
            reinterpret_cast<op::gelu::NAMESPACE::Descriptor **>(desc_ptr), \
            y_desc,                                                         \
            {x_desc})

    switch (handle->device) {
#ifdef ENABLE_CPU_API
        CTEATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        CTEATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_METAX_API
        CTEATE(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_ILUVATAR_API
        CTEATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CTEATE
}

__C infiniStatus_t infiniopGetGeluWorkspaceSize(infiniopGeluDescriptor_t desc, size_t *size) {
#define GET(CASE, NAMESPACE)                                                                \
    case CASE:                                                                              \
        *size = reinterpret_cast<op::gelu::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS;
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
#ifdef ENABLE_ILUVATAR_API
        GET(INFINI_DEVICE_ILUVATAR, nvidia);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET
}

__C infiniStatus_t infiniopGelu(
    infiniopGeluDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void *stream) {
#define CALCULATE(CASE, NAMESPACE)                                             \
    case CASE:                                                                 \
        return reinterpret_cast<const op::gelu::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, y, {x}, stream)

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
#ifdef ENABLE_ILUVATAR_API
        CALCULATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CALCULATE
}

__C infiniStatus_t infiniopDestroyGeluDescriptor(infiniopGeluDescriptor_t desc) {
#define DELETE(CASE, NAMESPACE)                                                 \
    case CASE:                                                                  \
        delete reinterpret_cast<const op::gelu::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;
    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        DELETE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        DELETE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_METAX_API
        DELETE(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_ILUVATAR_API
        DELETE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef DELETE
}
