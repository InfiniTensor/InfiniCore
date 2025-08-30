#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/sigmoid_backward.h"

#ifdef ENABLE_CPU_API
#include "cpu/sigmoid_backward_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
#include "nvidia/sigmoid_backward_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/sigmoid_backward_metax.h"
#endif

__C infiniStatus_t infiniopCreateSigmoidBackwardDescriptor(
    infiniopHandle_t handle,
    infiniopSigmoidBackwardDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t grad_input_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t grad_output_desc) {

#define CREATE(CASE, NAMESPACE)                                                         \
    case CASE:                                                                          \
        return op::sigmoid_backward::NAMESPACE::Descriptor::create(                     \
            handle,                                                                     \
            reinterpret_cast<op::sigmoid_backward::NAMESPACE::Descriptor **>(desc_ptr), \
            grad_input_desc,                                                            \
            {input_desc,                                                                \
             grad_output_desc})

    switch (handle->device) {

#ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CREATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_METAX_API
        CREATE(INFINI_DEVICE_METAX, metax);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__C infiniStatus_t infiniopGetSigmoidBackwardWorkspaceSize(infiniopSigmoidBackwardDescriptor_t desc, size_t *size) {

#define GET(CASE, NAMESPACE)                                                                            \
    case CASE:                                                                                          \
        *size = reinterpret_cast<op::sigmoid_backward::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        GET(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        GET(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_METAX_API
        GET(INFINI_DEVICE_METAX, metax);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopSigmoidBackward(
    infiniopSigmoidBackwardDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *grad_input,
    const void *input,
    const void *grad_output,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                         \
    case CASE:                                                                             \
        return reinterpret_cast<const op::sigmoid_backward::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, grad_input, {input, grad_output}, stream)

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CALCULATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_METAX_API
        CALCULATE(INFINI_DEVICE_METAX, metax);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__C infiniStatus_t
infiniopDestroySigmoidBackwardDescriptor(infiniopSigmoidBackwardDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                             \
    case CASE:                                                                              \
        delete reinterpret_cast<const op::sigmoid_backward::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        DELETE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        DELETE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        DELETE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_METAX_API
        DELETE(INFINI_DEVICE_METAX, metax);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DELETE
}