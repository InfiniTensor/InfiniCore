#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/repetition_penalty.h"

#ifdef ENABLE_CPU_API
#include "cpu/repetition_penalty_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API)
// TODO: Add NVIDIA implementation
#endif
#ifdef ENABLE_CAMBRICON_API
// TODO: Add Cambricon implementation
#endif
#ifdef ENABLE_METAX_API
#include "metax/repetition_penalty_metax.h"
#endif
#ifdef ENABLE_ASCEND_API
// TODO: Add Ascend implementation
#endif
#ifdef ENABLE_MOORE_API
// TODO: Add Moore implementation
#endif
#ifdef ENABLE_KUNLUN_API
// TODO: Add Kunlun implementation
#endif

__C infiniStatus_t infiniopCreateRepetitionPenaltyDescriptor(
    infiniopHandle_t handle,
    infiniopRepetitionPenaltyDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t logits_desc,
    infiniopTensorDescriptor_t mask_desc) {

#define CREATE(CASE, NAMESPACE)                                                      \
    case CASE:                                                                       \
        return op::repetition_penalty::NAMESPACE::Descriptor::create(                 \
            handle,                                                                  \
            reinterpret_cast<op::repetition_penalty::NAMESPACE::Descriptor **>(desc_ptr), \
            logits_desc,                                                             \
            mask_desc)

    switch (handle->device) {

#ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu);
#endif
// TODO: Add other backend implementations
// #ifdef ENABLE_NVIDIA_API
//         CREATE(INFINI_DEVICE_NVIDIA, nvidia);
// #endif
// #ifdef ENABLE_ILUVATAR_API
//         CREATE(INFINI_DEVICE_ILUVATAR, nvidia);
// #endif
// #ifdef ENABLE_HYGON_API
//         CREATE(INFINI_DEVICE_HYGON, nvidia);
// #endif
// #ifdef ENABLE_CAMBRICON_API
//         CREATE(INFINI_DEVICE_CAMBRICON, bang);
// #endif
#ifdef ENABLE_METAX_API
        CREATE(INFINI_DEVICE_METAX, metax);
#endif
// #ifdef ENABLE_ASCEND_API
//         CREATE(INFINI_DEVICE_ASCEND, ascend);
// #endif
// #ifdef ENABLE_MOORE_API
//         CREATE(INFINI_DEVICE_MOORE, moore);
// #endif
// #ifdef ENABLE_KUNLUN_API
//         CREATE(INFINI_DEVICE_KUNLUN, kunlun);
// #endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__C infiniStatus_t infiniopGetRepetitionPenaltyWorkspaceSize(
    infiniopRepetitionPenaltyDescriptor_t desc,
    size_t *size) {

#define GET(CASE, NAMESPACE)                                                          \
    case CASE: {                                                                     \
        using Ptr = const op::repetition_penalty::NAMESPACE::Descriptor *;            \
        *size = reinterpret_cast<Ptr>(desc)->workspaceSize();                        \
    }                                                                                \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        GET(INFINI_DEVICE_CPU, cpu);
#endif
// TODO: Add other backend implementations
// #ifdef ENABLE_NVIDIA_API
//         GET(INFINI_DEVICE_NVIDIA, nvidia);
// #endif
// #ifdef ENABLE_ILUVATAR_API
//         GET(INFINI_DEVICE_ILUVATAR, nvidia);
// #endif
// #ifdef ENABLE_HYGON_API
//         GET(INFINI_DEVICE_HYGON, nvidia);
// #endif
// #ifdef ENABLE_CAMBRICON_API
//         GET(INFINI_DEVICE_CAMBRICON, bang);
// #endif
#ifdef ENABLE_METAX_API
        GET(INFINI_DEVICE_METAX, metax);
#endif
// #ifdef ENABLE_ASCEND_API
//         GET(INFINI_DEVICE_ASCEND, ascend);
// #endif
// #ifdef ENABLE_MOORE_API
//         GET(INFINI_DEVICE_MOORE, moore);
// #endif
// #ifdef ENABLE_KUNLUN_API
//         GET(INFINI_DEVICE_KUNLUN, kunlun);
// #endif

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
    const void *mask,
    const float *repetition_penalties,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                      \
    case CASE:                                                                          \
        return reinterpret_cast<const op::repetition_penalty::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size,                                      \
                        logits, mask, repetition_penalties,                            \
                        stream)

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu);
#endif
// TODO: Add other backend implementations
// #ifdef ENABLE_NVIDIA_API
//         CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
// #endif
// #ifdef ENABLE_ILUVATAR_API
//         CALCULATE(INFINI_DEVICE_ILUVATAR, nvidia);
// #endif
// #ifdef ENABLE_HYGON_API
//         CALCULATE(INFINI_DEVICE_HYGON, nvidia);
// #endif
// #ifdef ENABLE_CAMBRICON_API
//         CALCULATE(INFINI_DEVICE_CAMBRICON, bang);
// #endif
#ifdef ENABLE_METAX_API
        CALCULATE(INFINI_DEVICE_METAX, metax);
#endif
// #ifdef ENABLE_ASCEND_API
//         CALCULATE(INFINI_DEVICE_ASCEND, ascend);
// #endif
// #ifdef ENABLE_MOORE_API
//         CALCULATE(INFINI_DEVICE_MOORE, moore);
// #endif
// #ifdef ENABLE_KUNLUN_API
//         CALCULATE(INFINI_DEVICE_KUNLUN, kunlun);
// #endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__C infiniStatus_t infiniopDestroyRepetitionPenaltyDescriptor(
    infiniopRepetitionPenaltyDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                          \
    case CASE:                                                                           \
        delete reinterpret_cast<const op::repetition_penalty::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        DELETE(INFINI_DEVICE_CPU, cpu);
#endif
// TODO: Add other backend implementations
// #ifdef ENABLE_NVIDIA_API
//         DELETE(INFINI_DEVICE_NVIDIA, nvidia);
// #endif
// #ifdef ENABLE_ILUVATAR_API
//         DELETE(INFINI_DEVICE_ILUVATAR, nvidia);
// #endif
// #ifdef ENABLE_HYGON_API
//         DELETE(INFINI_DEVICE_HYGON, nvidia);
// #endif
// #ifdef ENABLE_CAMBRICON_API
//         DELETE(INFINI_DEVICE_CAMBRICON, bang);
// #endif
#ifdef ENABLE_METAX_API
        DELETE(INFINI_DEVICE_METAX, metax);
#endif
// #ifdef ENABLE_ASCEND_API
//         DELETE(INFINI_DEVICE_ASCEND, ascend);
// #endif
// #ifdef ENABLE_MOORE_API
//         DELETE(INFINI_DEVICE_MOORE, moore);
// #endif
// #ifdef ENABLE_KUNLUN_API
//         DELETE(INFINI_DEVICE_KUNLUN, kunlun);
// #endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DELETE
}
