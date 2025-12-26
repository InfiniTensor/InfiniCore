#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/embedding.h"

#ifdef ENABLE_CPU_API
#include "cpu/embedding_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API)
#include "nvidia/embedding_nvidia.cuh"
#endif

__C infiniStatus_t infiniopCreateEmbeddingDescriptor(
    infiniopHandle_t handle,
    infiniopEmbeddingDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t weight_desc) {

#define CREATE(CASE, NAMESPACE)                                                  \
    case CASE:                                                                   \
        return op::embedding::NAMESPACE::Descriptor::create(                     \
            handle,                                                              \
            reinterpret_cast<op::embedding::NAMESPACE::Descriptor **>(desc_ptr), \
            output_desc,                                                         \
            input_desc,                                                          \
            weight_desc)

    switch (handle->device) {

#ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu);
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API)
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#if defined(ENABLE_ILUVATAR_API)
        CREATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#if defined(ENABLE_QY_API)
        CREATE(INFINI_DEVICE_QY, nvidia);
#endif
#if defined(ENABLE_HYGON_API)
        CREATE(INFINI_DEVICE_HYGON, nvidia);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__C infiniStatus_t infiniopEmbedding(
    infiniopEmbeddingDescriptor_t desc,
    void *output,
    const void *input,
    const void *weight,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                  \
    case CASE:                                                                      \
        return reinterpret_cast<const op::embedding::NAMESPACE::Descriptor *>(desc) \
            ->calculate(output, input, weight, stream)

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu);
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API)
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#if defined(ENABLE_ILUVATAR_API)
        CALCULATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#if defined(ENABLE_QY_API)
        CALCULATE(INFINI_DEVICE_QY, nvidia);
#endif
#if defined(ENABLE_HYGON_API)
        CALCULATE(INFINI_DEVICE_HYGON, nvidia);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__C infiniStatus_t infiniopDestroyEmbeddingDescriptor(infiniopEmbeddingDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                      \
    case CASE:                                                                       \
        delete reinterpret_cast<const op::embedding::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        DELETE(INFINI_DEVICE_CPU, cpu);
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API)
        DELETE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#if defined(ENABLE_ILUVATAR_API)
        DELETE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#if defined(ENABLE_QY_API)
        DELETE(INFINI_DEVICE_QY, nvidia);
#endif
#if defined(ENABLE_HYGON_API)
        DELETE(INFINI_DEVICE_HYGON, nvidia);
#endif
    }

#undef DELETE

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}
