#ifndef __INFINIOP_SCALED_MM_W8A8_API_H__
#define __INFINIOP_SCALED_MM_W8A8_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopScaledMmW8A8Descriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateScaledMmW8A8Descriptor(
    infiniopHandle_t handle,
    infiniopScaledMmW8A8Descriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t a_scales_desc,
    infiniopTensorDescriptor_t b_scales_desc,
    infiniopTensorDescriptor_t bias_desc,
    bool trans_weight);

__INFINI_C __export infiniStatus_t infiniopScaledMmW8A8(
    infiniopScaledMmW8A8Descriptor_t desc,
    void *out,
    const void *a,
    const void *b,
    const void *a_scales,
    const void *b_scales,
    const void *bias,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyScaledMmW8A8Descriptor(
    infiniopScaledMmW8A8Descriptor_t desc);

#endif
