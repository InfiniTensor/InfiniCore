#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/vdot.h"

#ifdef ENABLE_CPU_API
#include "cpu/vdot_cpu.h"
#endif

__C infiniStatus_t infiniopCreateVdotDescriptor(
    infiniopHandle_t handle,
    infiniopVdotDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {

#define CREATE(CASE, NAMESPACE)                                                     \
    case CASE:                                                                      \
        return op::vdot::NAMESPACE::Descriptor::create(                             \
            handle,                                                                 \
            reinterpret_cast<op::vdot::NAMESPACE::Descriptor **>(desc_ptr),         \
            out_desc,                                                               \
            a_desc, b_desc)

    switch (handle->device) {
#ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__C infiniStatus_t infiniopGetVdotWorkspaceSize(
    infiniopVdotDescriptor_t desc,
    size_t *size) {

#define GET(CASE, NAMESPACE)                                                                 \
    case CASE:                                                                               \
        *size = reinterpret_cast<op::vdot::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
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

__C infiniStatus_t infiniopVdot(
    infiniopVdotDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *a,
    const void *b,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                             \
    case CASE:                                                                 \
        return reinterpret_cast<const op::vdot::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, out, a, b, stream)

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__C infiniStatus_t infiniopDestroyVdotDescriptor(infiniopVdotDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                     \
    case CASE:                                                                      \
        delete reinterpret_cast<const op::vdot::NAMESPACE::Descriptor *>(desc);     \
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


