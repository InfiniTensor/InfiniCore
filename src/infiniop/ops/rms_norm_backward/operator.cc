#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/rms_norm_backward.h"

#ifdef ENABLE_CPU_API
#include "cpu/rms_norm_backward_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
#include "nvidia/rms_norm_backward_nvidia.h"
#endif
#ifdef ENABLE_METAX_API
#include "metax/rms_norm_backward_metax.h"
#endif

__C infiniStatus_t infiniopCreateRMSNormBackwardDescriptor(
    infiniopHandle_t handle,
    infiniopRMSNormBackwardDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t input_grad_desc,
    infiniopTensorDescriptor_t weight_grad_desc,
    infiniopTensorDescriptor_t output_grad_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t weight_desc,
    float epsilon) {

#define CREATE(CASE, NAMESPACE)                                                 \
    case CASE:                                                                  \
        return op::rms_norm_backward::NAMESPACE::Descriptor::create(            \
            handle,                                                             \
            reinterpret_cast<op::rms_norm_backward::NAMESPACE::Descriptor **>(desc_ptr), \
            input_grad_desc,                                                    \
            weight_grad_desc,                                                   \
            output_grad_desc,                                                   \
            input_desc,                                                         \
            weight_desc,                                                        \
            epsilon)

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
    }

#undef CREATE

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopGetRMSNormBackwardWorkspaceSize(infiniopRMSNormBackwardDescriptor_t desc, size_t *size) {

#define GET(CASE, NAMESPACE)                                                                          \
    case CASE:                                                                                        \
        *size = reinterpret_cast<op::rms_norm_backward::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
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
    }

#undef GET

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopRMSNormBackward(infiniopRMSNormBackwardDescriptor_t desc, void *workspace, size_t workspace_size,
                                           void *grad_input, void *grad_weight,
                                           const void *grad_output, const void *input, const void *weight, void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                             \
    case CASE:                                                                                 \
        return reinterpret_cast<op::rms_norm_backward::NAMESPACE::Descriptor *>(desc)->calculate( \
            workspace, workspace_size, grad_input, grad_weight, grad_output, input, weight, stream)

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
    }

#undef CALCULATE

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopDestroyRMSNormBackwardDescriptor(infiniopRMSNormBackwardDescriptor_t desc) {

#define DESTROY(CASE, NAMESPACE)                                                    \
    case CASE:                                                                      \
        delete reinterpret_cast<op::rms_norm_backward::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        DESTROY(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        DESTROY(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
    }

#undef DESTROY

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}