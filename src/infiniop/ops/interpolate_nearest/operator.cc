#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/interpolate_nearest.h"

#ifdef ENABLE_CPU_API
#include "cpu/interpolate_nearest_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API)
#include "nvidia/interpolate_nearest_nvidia.cuh"
#endif

__C infiniStatus_t infiniopCreateInterpolateNearestDescriptor(
    infiniopHandle_t handle,
    infiniopInterpolateNearestDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc) {

#define CREATE(CASE, NAMESPACE)                                                            \
    case CASE:                                                                             \
        return op::interpolate_nearest::NAMESPACE::Descriptor::create(                     \
            handle,                                                                        \
            reinterpret_cast<op::interpolate_nearest::NAMESPACE::Descriptor **>(desc_ptr), \
            output_desc,                                                                   \
            input_desc)

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
#ifdef ENABLE_QY_API
        CREATE(INFINI_DEVICE_QY, nvidia);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__C infiniStatus_t infiniopGetInterpolateNearestWorkspaceSize(infiniopInterpolateNearestDescriptor_t desc, size_t *size) {

#define GET(CASE, NAMESPACE)                                                                                     \
    case CASE:                                                                                                   \
        *size = reinterpret_cast<const op::interpolate_nearest::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS;

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
#ifdef ENABLE_QY_API
        GET(INFINI_DEVICE_QY, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopInterpolateNearest(
    infiniopInterpolateNearestDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                            \
    case CASE:                                                                                \
        return reinterpret_cast<const op::interpolate_nearest::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, output, input, stream)

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
#ifdef ENABLE_QY_API
        CALCULATE(INFINI_DEVICE_QY, nvidia);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__C infiniStatus_t
infiniopDestroyInterpolateNearestDescriptor(infiniopInterpolateNearestDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                                \
    case CASE:                                                                                 \
        delete reinterpret_cast<const op::interpolate_nearest::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

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
#ifdef ENABLE_QY_API
        DELETE(INFINI_DEVICE_QY, nvidia);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DELETE
}
