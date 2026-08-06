#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/scaled_mm_w8a8.h"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API) || defined(ENABLE_ALI_API)
#include "nvidia/scaled_mm_w8a8_nvidia.cuh"
#endif

__INFINI_C infiniStatus_t infiniopCreateScaledMmW8A8Descriptor(
    infiniopHandle_t handle,
    infiniopScaledMmW8A8Descriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t a_scales_desc,
    infiniopTensorDescriptor_t b_scales_desc,
    infiniopTensorDescriptor_t bias_desc,
    bool trans_weight) {
#define CREATE(CASE, NAMESPACE)                                                       \
    case CASE:                                                                        \
        return op::scaled_mm_w8a8::NAMESPACE::Descriptor::create(                     \
            handle,                                                                   \
            reinterpret_cast<op::scaled_mm_w8a8::NAMESPACE::Descriptor **>(desc_ptr), \
            out_desc, a_desc, b_desc, a_scales_desc, b_scales_desc, bias_desc,        \
            trans_weight)
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

__INFINI_C infiniStatus_t infiniopScaledMmW8A8(
    infiniopScaledMmW8A8Descriptor_t desc,
    void *out,
    const void *a,
    const void *b,
    const void *a_scales,
    const void *b_scales,
    const void *bias,
    void *stream) {
#define CALCULATE(CASE, NAMESPACE)                                                       \
    case CASE:                                                                           \
        return reinterpret_cast<const op::scaled_mm_w8a8::NAMESPACE::Descriptor *>(desc) \
            ->calculate(out, a, b, a_scales, b_scales, bias, stream)
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

__INFINI_C infiniStatus_t infiniopDestroyScaledMmW8A8Descriptor(
    infiniopScaledMmW8A8Descriptor_t desc) {
    delete reinterpret_cast<const op::scaled_mm_w8a8::nvidia::Descriptor *>(desc);
    return INFINI_STATUS_SUCCESS;
}
