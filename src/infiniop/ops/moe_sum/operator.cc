#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/moe_sum.h"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/moe_sum_nvidia.cuh"
#endif

__INFINI_C infiniStatus_t infiniopCreateMoeSumDescriptor(
    infiniopHandle_t handle,
    infiniopMoeSumDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc) {

#define CREATE(CASE, NAMESPACE)                                                \
    case CASE:                                                                 \
        return op::moe_sum::NAMESPACE::Descriptor::create(                     \
            handle,                                                            \
            reinterpret_cast<op::moe_sum::NAMESPACE::Descriptor **>(desc_ptr), \
            output_desc,                                                       \
            input_desc)

    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__INFINI_C infiniStatus_t infiniopGetMoeSumWorkspaceSize(
    infiniopMoeSumDescriptor_t desc,
    size_t *size) {

#define GET(CASE, NAMESPACE)                                                                   \
    case CASE:                                                                                 \
        *size = reinterpret_cast<op::moe_sum::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
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

__INFINI_C infiniStatus_t infiniopMoeSum(
    infiniopMoeSumDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                      \
    case CASE:                                                                          \
        return reinterpret_cast<op::moe_sum::NAMESPACE::Descriptor *>(desc)->calculate( \
            workspace, workspace_size, output, input, stream)

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__INFINI_C infiniStatus_t infiniopDestroyMoeSumDescriptor(
    infiniopMoeSumDescriptor_t desc) {

#define DESTROY(CASE, NAMESPACE)                                             \
    case CASE:                                                               \
        delete reinterpret_cast<op::moe_sum::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DESTROY
}
