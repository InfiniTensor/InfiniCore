#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/layer_norm_backward.h"

#ifdef ENABLE_CPU_API
#include "cpu/layer_norm_backward_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
#include "nvidia/layer_norm_backward_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/layer_norm_backward_metax.h"
#endif

__C infiniStatus_t infiniopCreateLayerNormBackwardDescriptor(
    infiniopHandle_t handle,
    infiniopLayerNormBackwardDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t grad_input_desc,
    infiniopTensorDescriptor_t grad_weight_desc,
    infiniopTensorDescriptor_t grad_bias_desc,
    infiniopTensorDescriptor_t grad_output_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t input_standardization_desc,
    infiniopTensorDescriptor_t input_std_deviation_desc
) {
#define CREATE(CASE, NAMESPACE)                                                                     \
    case CASE:                                                                                      \
        return op::layer_norm_backward::NAMESPACE::Descriptor::create(                              \
            handle,                                                                                 \
            reinterpret_cast<op::layer_norm_backward::NAMESPACE::Descriptor **>(desc_ptr),          \
            grad_input_desc,                                                                        \
            grad_weight_desc,                                                                       \
            grad_bias_desc,                                                                         \
            grad_output_desc,                                                                       \
            weight_desc,                                                                            \
            input_standardization_desc,                                                             \
            input_std_deviation_desc                                                                \
        )

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

__C infiniStatus_t infiniopGetLayerNormBackwardWorkspaceSize(infiniopLayerNormBackwardDescriptor_t desc, size_t *size) {
#define GET(CASE, NAMESPACE)                                                                                  \
    case CASE:                                                                                                \
        *size = reinterpret_cast<op::layer_norm_backward::NAMESPACE::Descriptor *>(desc)->workspaceSize();    \
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

__C infiniStatus_t infiniopLayerNormBackward(
    infiniopLayerNormBackwardDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void * grad_input,
    void * grad_weight,
    void * grad_bias,
    const void * grad_output,
    const void * weight,
    const void * input_standardization,
    const void * input_std_deviation,                
    void *stream
) {

#define CALCULATE(CASE, NAMESPACE)                                                                            \
    case CASE:                                                                                                \
        return reinterpret_cast<const op::layer_norm_backward::NAMESPACE::Descriptor *>(desc)->calculate(     \
            workspace,                                                                                        \
            workspace_size,                                                                                   \
            grad_input,                                                                                       \
            grad_weight,                                                                                      \
            grad_bias,                                                                                        \
            grad_output,                                                                                      \
            weight,                                                                                           \
            input_standardization,                                                                            \
            input_std_deviation,                                                                              \
            stream                                                                                            \
        )

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
infiniopDestroyLayerNormBackwardDescriptor(infiniopLayerNormBackwardDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                                     \
    case CASE:                                                                                      \
        delete reinterpret_cast<const op::layer_norm_backward::NAMESPACE::Descriptor *>(desc);      \
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
