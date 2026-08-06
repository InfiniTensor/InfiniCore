#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/w4a8_group_gemm.h"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API) || defined(ENABLE_ALI_API)
#include "nvidia/w4a8_group_gemm_nvidia.cuh"
#endif

__INFINI_C infiniStatus_t infiniopCreateW4A8GroupGemmDescriptor(
    infiniopHandle_t handle,
    infiniopW4A8GroupGemmDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t input_scale_desc,
    infiniopTensorDescriptor_t weight_scale_desc,
    infiniopTensorDescriptor_t tokens_per_experts_desc,
    infiniopTensorDescriptor_t sorted_token_ids_desc,
    infiniopTensorDescriptor_t bias_desc,
    bool trans_weight) {
#define CREATE(CASE, NAMESPACE)                                                \
    case CASE:                                                                 \
        return op::w4a8_group_gemm::NAMESPACE::Descriptor::create(             \
            handle,                                                            \
            reinterpret_cast<op::w4a8_group_gemm::NAMESPACE::Descriptor **>(   \
                desc_ptr),                                                     \
            out_desc, input_desc, weight_desc, input_scale_desc,               \
            weight_scale_desc, tokens_per_experts_desc, sorted_token_ids_desc, \
            bias_desc, trans_weight)
    switch (handle->device) {
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

__INFINI_C infiniStatus_t infiniopW4A8GroupGemm(
    infiniopW4A8GroupGemmDescriptor_t desc,
    void *out,
    const void *input,
    const void *weight,
    const void *input_scale,
    const void *weight_scale,
    const void *tokens_per_experts,
    const void *sorted_token_ids,
    const void *bias,
    void *stream) {
#define CALCULATE(CASE, NAMESPACE)                                           \
    case CASE:                                                               \
        return reinterpret_cast<                                             \
                   const op::w4a8_group_gemm::NAMESPACE::Descriptor *>(desc) \
            ->calculate(out, input, weight, input_scale, weight_scale,       \
                        tokens_per_experts, sorted_token_ids, bias, stream)
    switch (desc->device_type) {
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

__INFINI_C infiniStatus_t infiniopDestroyW4A8GroupGemmDescriptor(
    infiniopW4A8GroupGemmDescriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                       \
    case CASE:                                                         \
        delete reinterpret_cast<                                       \
            const op::w4a8_group_gemm::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS
    switch (desc->device_type) {
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
