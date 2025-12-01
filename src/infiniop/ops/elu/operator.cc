#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/elu.h"

#ifdef ENABLE_CPU_API
#include "cpu/elu_cpu.h"
#endif

__C infiniStatus_t infiniopCreateEluDescriptor(
    infiniopHandle_t handle,
    infiniopEluDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output,
    infiniopTensorDescriptor_t input,
    float alpha) {

#define CREATE(CASE, NAMESPACE)                                            \
    case CASE:                                                             \
        return op::elu::NAMESPACE::Descriptor::create(                     \
            handle,                                                        \
            reinterpret_cast<op::elu::NAMESPACE::Descriptor **>(desc_ptr), \
            output,                                                        \
            {input},                                                       \
            alpha);

    switch (handle->device) {
#ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__C infiniStatus_t infiniopGetEluWorkspaceSize(
    infiniopEluDescriptor_t desc,
    size_t *size) {

#define GET(CASE, NAMESPACE)                                                             \
    case CASE:                                                                           \
        *size                                                                            \
            = reinterpret_cast<op::elu::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        GET(INFINI_DEVICE_CPU, cpu)
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET
}

__C infiniStatus_t infiniopElu(
    infiniopEluDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                            \
    case CASE:                                                                \
        return reinterpret_cast<const op::elu::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, output, {input}, stream)

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__C infiniStatus_t infiniopDestroyEluDescriptor(infiniopEluDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                \
    case CASE:                                                                 \
        delete reinterpret_cast<const op::elu::NAMESPACE::Descriptor *>(desc); \
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
