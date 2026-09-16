#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/w4a8_moe_shuffle.h"

#if defined ENABLE_NVIDIA_API || defined ENABLE_HYGON_API
#include "nvidia/w4a8_moe_shuffle_nvidia.cuh"
#endif

__INFINI_C infiniStatus_t infiniopCreateW4A8MoeShuffleDescriptor(
    infiniopHandle_t handle,
    infiniopW4A8MoeShuffleDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc) {
#define CREATE(CASE, NAMESPACE)                                                                 \
    case CASE:                                                                                  \
        return op::w4a8_moe_shuffle::NAMESPACE::Descriptor::create(                             \
            handle, reinterpret_cast<op::w4a8_moe_shuffle::NAMESPACE::Descriptor **>(desc_ptr), \
            output_desc, input_desc)
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

__INFINI_C infiniStatus_t infiniopW4A8MoeShuffle(
    infiniopW4A8MoeShuffleDescriptor_t desc,
    void *output,
    const void *input,
    void *stream) {
#define CALCULATE(CASE, NAMESPACE)                                                    \
    case CASE:                                                                        \
        return reinterpret_cast<const op::w4a8_moe_shuffle::NAMESPACE::Descriptor *>( \
                   desc)                                                              \
            ->calculate(output, input, stream)
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

__INFINI_C infiniStatus_t infiniopDestroyW4A8MoeShuffleDescriptor(
    infiniopW4A8MoeShuffleDescriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                                      \
    case CASE:                                                                        \
        delete reinterpret_cast<const op::w4a8_moe_shuffle::NAMESPACE::Descriptor *>( \
            desc);                                                                    \
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
