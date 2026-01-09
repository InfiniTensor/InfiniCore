#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/repetition_penalty.h"

#ifdef ENABLE_CPU_API
#include "cpu/repetition_penalty_cpu.h"
#endif
#ifdef ENABLE_METAX_API
#include "metax/repetition_penalty_metax.h"
#endif

__C infiniStatus_t infiniopCreateRepetitionPenaltyDescriptor(
    infiniopHandle_t handle,
    infiniopRepetitionPenaltyDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t logits_desc) {

#define CREATE(CASE, NAMESPACE)                                                      \
    case CASE:                                                                       \
        return op::repetition_penalty::NAMESPACE::Descriptor::create(                \
            handle,                                                                  \
            reinterpret_cast<op::repetition_penalty::NAMESPACE::Descriptor **>(desc_ptr), \
            logits_desc)

    switch (handle->device) {

#ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_METAX_API
        CREATE(INFINI_DEVICE_METAX, metax);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__C infiniStatus_t infiniopGetRepetitionPenaltyWorkspaceSize(
    infiniopRepetitionPenaltyDescriptor_t desc,
    size_t *size) {

#define GET(CASE, NAMESPACE)                                                         \
    case CASE: {                                                                     \
        using Ptr = const op::repetition_penalty::NAMESPACE::Descriptor *;           \
        *size = reinterpret_cast<Ptr>(desc)->minWorkspaceSize();                     \
    }                                                                                 \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        GET(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_METAX_API
        GET(INFINI_DEVICE_METAX, metax);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef GET
}

__C infiniStatus_t infiniopApplyRepetitionPenalty(
    infiniopRepetitionPenaltyDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *logits,
    const float *repetition_penalties,
    const uint32_t *token_indices,
    const size_t *token_offsets,
    size_t total_indices,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                   \
    case CASE:                                                                       \
        return reinterpret_cast<const op::repetition_penalty::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size,                                   \
                        logits, repetition_penalties,                                \
                        token_indices, token_offsets,                                \
                        total_indices,                                               \
                        stream)

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_METAX_API
        CALCULATE(INFINI_DEVICE_METAX, metax);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__C infiniStatus_t infiniopDestroyRepetitionPenaltyDescriptor(
    infiniopRepetitionPenaltyDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                      \
    case CASE:                                                                       \
        delete reinterpret_cast<const op::repetition_penalty::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        DELETE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_METAX_API
        DELETE(INFINI_DEVICE_METAX, metax);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DELETE
}
