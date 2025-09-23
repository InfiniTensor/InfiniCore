#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/logsoftmax.h"

#ifdef ENABLE_CPU_API
#include "cpu/logsoftmax_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
#include "nvidia/logsoftmax_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
// #include "metax/logsoftmax_metax.h"
#endif
#ifdef ENABLE_ASCEND_API
// #include "ascend/logsoftmax_ascend.h"
#endif

__C infiniStatus_t infiniopCreateLogSoftmaxDescriptor(
    infiniopHandle_t handle,
    infiniopLogSoftmaxDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc) {

#define CREATE(CASE, NAMESPACE)                                                   \
    case CASE:                                                                    \
        return op::logsoftmax::NAMESPACE::Descriptor::create(                     \
            handle,                                                               \
            reinterpret_cast<op::logsoftmax::NAMESPACE::Descriptor **>(desc_ptr), \
            y_desc,                                                               \
            x_desc);

    switch (handle->device) {
#ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu)
#endif
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_ILUVATAR_API
        // CREATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_METAX_API
        // CREATE(INFINI_DEVICE_METAX, metax)
#endif
#ifdef ENABLE_ASCEND_API
        // CREATE(INFINI_DEVICE_ASCEND, ascend)
#endif
    }
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopGetLogSoftmaxWorkspaceSize(infiniopLogSoftmaxDescriptor_t desc, size_t *size) {

#define GET(CASE, NAMESPACE)                                                                      \
    case CASE:                                                                                    \
        *size = reinterpret_cast<op::logsoftmax::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        GET(INFINI_DEVICE_CPU, cpu)
#endif
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_ILUVATAR_API
        // GET(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_METAX_API
        // GET(INFINI_DEVICE_METAX, metax)
#endif
#ifdef ENABLE_ASCEND_API
        // GET(INFINI_DEVICE_ASCEND, ascend)
#endif
    }
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopLogSoftmax(
    infiniopLogSoftmaxDescriptor_t desc,
    void *workspace, size_t workspace_size,
    void *y,
    const void *x,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                         \
    case CASE:                                                                             \
        return reinterpret_cast<op::logsoftmax::NAMESPACE::Descriptor *>(desc)->calculate( \
            workspace, workspace_size, y, x, stream);

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu)
#endif
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_ILUVATAR_API
        // CALCULATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_METAX_API
        // CALCULATE(INFINI_DEVICE_METAX, metax)
#endif
#ifdef ENABLE_ASCEND_API
        // CALCULATE(INFINI_DEVICE_ASCEND, ascend)
#endif
    }
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopDestroyLogSoftmaxDescriptor(infiniopLogSoftmaxDescriptor_t desc) {

#define DESTROY(CASE, NAMESPACE)                                                \
    case CASE:                                                                  \
        delete reinterpret_cast<op::logsoftmax::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        DESTROY(INFINI_DEVICE_CPU, cpu)
#endif
#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_ILUVATAR_API
        // DESTROY(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_METAX_API
        // DESTROY(INFINI_DEVICE_METAX, metax)
#endif
#ifdef ENABLE_ASCEND_API
        // DESTROY(INFINI_DEVICE_ASCEND, ascend)
#endif
    }
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}
