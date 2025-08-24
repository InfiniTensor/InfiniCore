#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/batch_norm.h"

#ifdef ENABLE_CPU_API
#include "cpu/batch_norm_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
#include "nvidia/batch_norm_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/batch_norm_metax.h"
#endif

__C infiniStatus_t infiniopCreateBatchNormDescriptor(
    infiniopHandle_t handle,
    infiniopBatchNormDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t running_mean_desc,
    infiniopTensorDescriptor_t running_var_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t bias_desc,
    float momentum,
    float eps
) {
#define CREATE(CASE, NAMESPACE)                                                           \
    case CASE:                                                                            \
        return op::batch_norm::NAMESPACE::Descriptor::create(                             \
            handle,                                                                       \
            reinterpret_cast<op::batch_norm::NAMESPACE::Descriptor **>(desc_ptr),         \
            output_desc,                                                                  \
            running_mean_desc,                                                            \
            running_var_desc,                                                             \
            input_desc,                                                                   \
            weight_desc,                                                                  \
            bias_desc,                                                                    \
            momentum,                                                                     \
            eps                                                                           \
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

__C infiniStatus_t infiniopGetBatchNormWorkspaceSize(infiniopBatchNormDescriptor_t desc, size_t *size) {
#define GET(CASE, NAMESPACE)                                                                        \
    case CASE:                                                                                      \
        *size = reinterpret_cast<op::batch_norm::NAMESPACE::Descriptor *>(desc)->workspaceSize();   \
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

__C infiniStatus_t infiniopBatchNorm(
    infiniopBatchNormDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void * output,
    void * running_mean,
    void * running_var,
    const void * input,
    const void * weight,
    const void * bias,                
    void *stream
) {

#define CALCULATE(CASE, NAMESPACE)                                                                  \
    case CASE:                                                                                      \
        return reinterpret_cast<const op::batch_norm::NAMESPACE::Descriptor *>(desc)->calculate(    \
            workspace,                                                                              \
            workspace_size,                                                                         \
            output,                                                                                 \
            running_mean,                                                                           \
            running_var,                                                                            \
            input,                                                                                  \
            weight,                                                                                 \
            bias,                                                                                   \
            stream                                                                                  \
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
infiniopDestroyBatchNormDescriptor(infiniopBatchNormDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                           \
    case CASE:                                                                            \
        delete reinterpret_cast<const op::batch_norm::NAMESPACE::Descriptor *>(desc);     \
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
