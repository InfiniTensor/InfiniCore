#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/conv_backward.h"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
#include "nvidia/conv_backward_nvidia.cuh"
#endif

__C infiniStatus_t infiniopCreateConvBackwardDescriptor(
    infiniopHandle_t handle,
    infiniopConvBackwardDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t grad_output_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t bias_desc,
    void *pads,
    void *strides,
    void *dilations,
    size_t groups) {
#define CREATE(CASE, NAMESPACE)                                                              \
    case CASE:                                                                               \
        return op::conv_backward::NAMESPACE::Descriptor::create(                             \
            handle, reinterpret_cast<op::conv_backward::NAMESPACE::Descriptor **>(desc_ptr), \
            grad_output_desc, input_desc, weight_desc, bias_desc, pads, strides, dilations, groups)

    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CREATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CREATE
}

__C infiniStatus_t infiniopGetConvBackwardWorkspaceSize(
    infiniopConvBackwardDescriptor_t desc, size_t *size) {
#define GET(CASE, NAMESPACE)                                                                               \
    case CASE:                                                                                             \
        *size = reinterpret_cast<const op::conv_backward::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        GET(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET
}

__C infiniStatus_t infiniopConvBackward(
    infiniopConvBackwardDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *grad_input,
    void *grad_weight,
    void *grad_bias,
    const void *grad_output,
    const void *input,
    const void *weight,
    void *stream) {
#define CALCULATE(CASE, NAMESPACE)                                                      \
    case CASE:                                                                          \
        return reinterpret_cast<const op::conv_backward::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, grad_input, grad_weight, grad_bias, grad_output, input, weight, stream)

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CALCULATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CALCULATE
}

__C infiniStatus_t infiniopDestroyConvBackwardDescriptor(infiniopConvBackwardDescriptor_t desc) {
#define DELETE(CASE, NAMESPACE)                                                          \
    case CASE:                                                                           \
        delete reinterpret_cast<const op::conv_backward::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        DELETE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        DELETE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef DELETE
}
