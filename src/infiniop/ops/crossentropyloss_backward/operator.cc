#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/crossentropyloss_backward.h"

#ifdef ENABLE_CPU_API
#include "cpu/crossentropyloss_backward_cpu.h"
#endif

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
#include "nvidia/crossentropyloss_backward_nvidia.cuh"
#endif

#ifdef ENABLE_METAX_API
#include "metax/crossentropyloss_backward_metax.h"
#endif

#ifdef ENABLE_KUNLUN_API
#include "kunlun/crossentropyloss_backward_kunlun.h"
#endif

__C infiniStatus_t infiniopCreateCrossEntropyLossBackwardDescriptor(
    infiniopHandle_t handle,
    infiniopCrossEntropyLossBackwardDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t grad_logits_desc,
    infiniopTensorDescriptor_t probs_desc,
    infiniopTensorDescriptor_t target_desc) {

#define CREATE(CASE, NAMESPACE)                                                                     \
    case CASE:                                                                                      \
        return op::crossentropyloss_backward::NAMESPACE::Descriptor::create(                        \
            handle,                                                                                 \
            reinterpret_cast<op::crossentropyloss_backward::NAMESPACE::Descriptor **>(desc_ptr),   \
            grad_logits_desc,                                                                       \
            {probs_desc, target_desc})

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

#ifdef ENABLE_KUNLUN_API
        CREATE(INFINI_DEVICE_KUNLUN, kunlun);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__C infiniStatus_t infiniopGetCrossEntropyLossBackwardWorkspaceSize(infiniopCrossEntropyLossBackwardDescriptor_t desc, size_t *size) {
#define GET_WORKSPACE_SIZE(CASE, NAMESPACE)                                         \
    case CASE:                                                                      \
        *size = reinterpret_cast<op::crossentropyloss_backward::NAMESPACE::Descriptor *>(desc) \
            ->workspaceSize(); \
        return INFINI_STATUS_SUCCESS

    switch (reinterpret_cast<InfiniopDescriptor *>(desc)->device_type) {

#ifdef ENABLE_CPU_API
        GET_WORKSPACE_SIZE(INFINI_DEVICE_CPU, cpu);
#endif

#ifdef ENABLE_NVIDIA_API
        GET_WORKSPACE_SIZE(INFINI_DEVICE_NVIDIA, nvidia);
#endif

#ifdef ENABLE_ILUVATAR_API
        GET_WORKSPACE_SIZE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif

#ifdef ENABLE_METAX_API
        GET_WORKSPACE_SIZE(INFINI_DEVICE_METAX, metax);
#endif

#ifdef ENABLE_KUNLUN_API
        GET_WORKSPACE_SIZE(INFINI_DEVICE_KUNLUN, kunlun);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef GET_WORKSPACE_SIZE
}

__C infiniStatus_t infiniopCrossEntropyLossBackward(
    infiniopCrossEntropyLossBackwardDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *grad_logits,
    const void *probs,
    const void *target,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                                  \
    case CASE:                                                                                      \
        return reinterpret_cast<op::crossentropyloss_backward::NAMESPACE::Descriptor *>(desc)      \
            ->calculate(workspace, workspace_size, grad_logits, {probs, target}, stream)

    switch (reinterpret_cast<InfiniopDescriptor *>(desc)->device_type) {

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

#ifdef ENABLE_KUNLUN_API
        CALCULATE(INFINI_DEVICE_KUNLUN, kunlun);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__C infiniStatus_t
infiniopDestroyCrossEntropyLossBackwardDescriptor(infiniopCrossEntropyLossBackwardDescriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                                                    \
    case CASE:                                                                                      \
        delete reinterpret_cast<op::crossentropyloss_backward::NAMESPACE::Descriptor *>(desc);     \
        return INFINI_STATUS_SUCCESS

    switch (reinterpret_cast<InfiniopDescriptor *>(desc)->device_type) {

#ifdef ENABLE_CPU_API
        DESTROY(INFINI_DEVICE_CPU, cpu);
#endif

#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia);
#endif

#ifdef ENABLE_ILUVATAR_API
        DESTROY(INFINI_DEVICE_ILUVATAR, nvidia);
#endif

#ifdef ENABLE_METAX_API
        DESTROY(INFINI_DEVICE_METAX, metax);
#endif

#ifdef ENABLE_KUNLUN_API
        DESTROY(INFINI_DEVICE_KUNLUN, kunlun);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DESTROY
}