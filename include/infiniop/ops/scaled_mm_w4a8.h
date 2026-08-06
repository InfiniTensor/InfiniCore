#ifndef __INFINIOP_SCALED_MM_W4A8_API_H__
#define __INFINIOP_SCALED_MM_W4A8_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopScaledMmW4A8Descriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateScaledMmW4A8Descriptor(
    infiniopHandle_t handle,
    infiniopScaledMmW4A8Descriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t a_scales_desc,
    infiniopTensorDescriptor_t b_scales_desc,
    infiniopTensorDescriptor_t bias_desc,
    bool trans_weight);

__INFINI_C __export infiniStatus_t infiniopScaledMmW4A8(
    infiniopScaledMmW4A8Descriptor_t desc,
    void *out,
    const void *a,
    const void *b,
    const void *a_scales,
    const void *b_scales,
    const void *bias,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyScaledMmW4A8Descriptor(
    infiniopScaledMmW4A8Descriptor_t desc);

__INFINI_C __export infiniStatus_t infiniopPrepareGlmW4A16Awq(
    infiniopHandle_t handle,
    infiniopTensorDescriptor_t qweight_desc,
    infiniopTensorDescriptor_t qzeros_desc,
    infiniopTensorDescriptor_t scales_desc,
    infiniopTensorDescriptor_t checkpoint_weight_desc,
    infiniopTensorDescriptor_t channel_scales_desc,
    void *qweight,
    void *qzeros,
    void *scales,
    const void *checkpoint_weight,
    const void *channel_scales,
    void *stream);

#endif
