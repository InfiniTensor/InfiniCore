#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/select_last_token_hidden.h"

#ifdef ENABLE_CPU_API
#include "cpu/select_last_token_hidden_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API) || defined(ENABLE_ALI_API)
#include "nvidia/select_last_token_hidden_nvidia.cuh"
#endif

__INFINI_C infiniStatus_t infiniopCreateSelectLastTokenHiddenDescriptor(
    infiniopHandle_t handle,
    infiniopSelectLastTokenHiddenDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t hidden_states_desc,
    infiniopTensorDescriptor_t input_offsets_desc) {

#define CREATE(CASE, NAMESPACE)                                                                 \
    case CASE:                                                                                  \
        return op::select_last_token_hidden::NAMESPACE::Descriptor::create(                     \
            handle,                                                                             \
            reinterpret_cast<op::select_last_token_hidden::NAMESPACE::Descriptor **>(desc_ptr), \
            output_desc, hidden_states_desc, input_offsets_desc)

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
#ifdef ENABLE_HYGON_API
        CREATE(INFINI_DEVICE_HYGON, nvidia);
#endif
#ifdef ENABLE_ALI_API
        CREATE(INFINI_DEVICE_ALI, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__INFINI_C infiniStatus_t infiniopSelectLastTokenHidden(
    infiniopSelectLastTokenHiddenDescriptor_t desc,
    void *output,
    const void *hidden_states,
    const void *input_offsets,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                                 \
    case CASE:                                                                                     \
        return reinterpret_cast<const op::select_last_token_hidden::NAMESPACE::Descriptor *>(desc) \
            ->calculate(output, hidden_states, input_offsets, stream)

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
#ifdef ENABLE_HYGON_API
        CALCULATE(INFINI_DEVICE_HYGON, nvidia);
#endif
#ifdef ENABLE_ALI_API
        CALCULATE(INFINI_DEVICE_ALI, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__INFINI_C infiniStatus_t infiniopDestroySelectLastTokenHiddenDescriptor(
    infiniopSelectLastTokenHiddenDescriptor_t desc) {

#define DESTROY(CASE, NAMESPACE)                                                                    \
    case CASE:                                                                                      \
        delete reinterpret_cast<const op::select_last_token_hidden::NAMESPACE::Descriptor *>(desc); \
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
#ifdef ENABLE_QY_API
        DESTROY(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        DESTROY(INFINI_DEVICE_HYGON, nvidia);
#endif
#ifdef ENABLE_ALI_API
        DESTROY(INFINI_DEVICE_ALI, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DESTROY
}
