#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/erfc.h"

#ifdef ENABLE_CPU_API
#include "cpu/erfc_cpu.h"
#endif

__C infiniStatus_t infiniopCreateErfcDescriptor(
    infiniopHandle_t handle,
    infiniopErfcDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc) {

#define CREATE(CASE, NAMESPACE)                                                \
    case CASE:                                                                 \
        return op::erfc::NAMESPACE::Descriptor::create(                        \
            handle,                                                            \
            reinterpret_cast<op::erfc::NAMESPACE::Descriptor **>(desc_ptr),    \
            y_desc,                                                            \
            {x_desc})

    switch (handle->device) {

#ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__C infiniStatus_t infiniopGetErfcWorkspaceSize(infiniopErfcDescriptor_t desc, size_t *size) {

#define GET(CASE, NAMESPACE)                                                                   \
    case CASE:                                                                                 \
        *size = reinterpret_cast<op::erfc::NAMESPACE::Descriptor *>(desc)->workspaceSize();    \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        GET(INFINI_DEVICE_CPU, cpu)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopErfc(
    infiniopErfcDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                \
    case CASE:                                                                    \
        return reinterpret_cast<const op::erfc::NAMESPACE::Descriptor *>(desc)    \
            ->calculate(workspace, workspace_size, y, {x}, stream)

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__C infiniStatus_t
infiniopDestroyErfcDescriptor(infiniopErfcDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                    \
    case CASE:                                                                     \
        delete reinterpret_cast<const op::erfc::NAMESPACE::Descriptor *>(desc);    \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        DELETE(INFINI_DEVICE_CPU, cpu);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DELETE
}
