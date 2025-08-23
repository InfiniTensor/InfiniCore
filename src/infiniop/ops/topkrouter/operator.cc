#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/topkrouter.h"

#ifdef ENABLE_CPU_API
#include "cpu/topkrouter_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API)
#include "nvidia/topkrouter_nvidia.cuh"
#endif

__C infiniStatus_t infiniopCreateTopkrouterDescriptor(
    infiniopHandle_t handle,
    infiniopTopkrouterDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t correction_bias_desc) {

#define CREATE(CASE, NAMESPACE)                                                   \
    case CASE:                                                                    \
        return op::topkrouter::NAMESPACE::Descriptor::create(                     \
            handle,                                                               \
            reinterpret_cast<op::topkrouter::NAMESPACE::Descriptor **>(desc_ptr), \
            x_desc, correction_bias_desc)

    switch (handle->device) {
#ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    }

#undef CREATE

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopGetTopkrouterWorkspaceSize(infiniopTopkrouterDescriptor_t desc, size_t *size) {

#define GET(CASE, NAMESPACE)                                                                      \
    case CASE:                                                                                    \
        *size = reinterpret_cast<op::topkrouter::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        GET(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    }

#undef GET

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopTopkrouter(infiniopTopkrouterDescriptor_t desc, void *workspace, size_t workspace_size,
                                      void *values, void *indices, void *x, void *correction_bias, float routed_scaling_factor, size_t topk, void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                         \
    case CASE:                                                                             \
        return reinterpret_cast<op::topkrouter::NAMESPACE::Descriptor *>(desc)->calculate( \
            workspace, workspace_size, (float *)values, (int *)indices, x, (float *)correction_bias, routed_scaling_factor, topk, stream)

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    }

#undef CALCULATE

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopDestroyTopkrouterDescriptor(infiniopTopkrouterDescriptor_t desc) {

#define DESTROY(CASE, NAMESPACE)                                                \
    case CASE:                                                                  \
        delete reinterpret_cast<op::topkrouter::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        DESTROY(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    }

#undef DESTROY

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}
