#include "../../handle.h"
#include "../../operator.h"
#include "infiniop/ops/timestep_embedding.h"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_HYGON_API)
#include "nvidia/timestep_embedding_nvidia.cuh"
#endif

__INFINI_C infiniStatus_t infiniopCreateTimestepEmbeddingDescriptor(
    infiniopHandle_t handle,
    infiniopTimestepEmbeddingDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t timestep_desc) {

#define CREATE(CASE, NAMESPACE)                                                   \
    case CASE:                                                                    \
        return op::timestep_embedding::NAMESPACE::Descriptor::create(             \
            handle,                                                               \
            reinterpret_cast<op::timestep_embedding::NAMESPACE::Descriptor **>(   \
                desc_ptr),                                                        \
            output_desc,                                                          \
            timestep_desc)

    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        CREATE(INFINI_DEVICE_HYGON, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__INFINI_C infiniStatus_t infiniopTimestepEmbedding(
    infiniopTimestepEmbeddingDescriptor_t desc,
    void *output,
    const void *timestep,
    float max_period,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                              \
    case CASE:                                                                  \
        return reinterpret_cast<                                               \
                   const op::timestep_embedding::NAMESPACE::Descriptor *>(desc) \
            ->calculate(output, timestep, max_period, stream)

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        CALCULATE(INFINI_DEVICE_HYGON, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__INFINI_C infiniStatus_t infiniopDestroyTimestepEmbeddingDescriptor(
    infiniopTimestepEmbeddingDescriptor_t desc) {

#define DESTROY(CASE, NAMESPACE)                                               \
    case CASE:                                                                 \
        delete reinterpret_cast<                                               \
            const op::timestep_embedding::NAMESPACE::Descriptor *>(desc);      \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        DESTROY(INFINI_DEVICE_HYGON, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DESTROY
}
