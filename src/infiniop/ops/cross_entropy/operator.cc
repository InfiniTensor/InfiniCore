#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/cross_entropy.h"

#ifdef ENABLE_CPU_API
#include "cpu/cross_entropy_cpu.h"
#endif

__C infiniStatus_t infiniopCreateCrossEntropyDescriptor(
    infiniopHandle_t handle,
    infiniopCrossEntropyDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t target_desc) {
#define CREATE(CASE, NAMESPACE)                                                  \
    case CASE:                                                                   \
        return op::cross_entropy::NAMESPACE::Descriptor::create(                 \
            handle,                                                              \
            reinterpret_cast<op::cross_entropy::NAMESPACE::Descriptor **>(desc_ptr), \
            y_desc, x_desc, target_desc)

    switch (handle->device) {
#ifdef ENABLE_CPU_API
    CREATE(INFINI_DEVICE_CPU, cpu);
#endif
        default:
            return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CREATE
}

__C infiniStatus_t infiniopGetCrossEntropyWorkspaceSize(
    infiniopCrossEntropyDescriptor_t desc, size_t *size) {
#define GET(CASE, NAMESPACE)                                                     \
    case CASE:                                                                   \
        *size = reinterpret_cast<op::cross_entropy::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
    GET(INFINI_DEVICE_CPU, cpu);
#endif
        default:
            return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET
}

__C infiniStatus_t infiniopCrossEntropy(
    infiniopCrossEntropyDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    const void *target,
    void *stream) {
#define CALCULATE(CASE, NAMESPACE)                                               \
    case CASE:                                                                   \
        return reinterpret_cast<op::cross_entropy::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, y, x, target, stream)

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
    CALCULATE(INFINI_DEVICE_CPU, cpu);
#endif
        default:
            return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CALCULATE
}

__C infiniStatus_t infiniopDestroyCrossEntropyDescriptor(
    infiniopCrossEntropyDescriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                                 \
    case CASE:                                                                   \
        delete reinterpret_cast<op::cross_entropy::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
    DESTROY(INFINI_DEVICE_CPU, cpu);
#endif
        default:
            return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef DESTROY
}