#ifndef __INFINIOP_DSV4_LINEAR_BF16_FP32_API_H__
#define __INFINIOP_DSV4_LINEAR_BF16_FP32_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopDsv4LinearBf16Fp32Descriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateDsv4LinearBf16Fp32Descriptor(
    infiniopHandle_t handle,
    infiniopDsv4LinearBf16Fp32Descriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t w_desc);

__INFINI_C __export infiniStatus_t infiniopGetDsv4LinearBf16Fp32WorkspaceSize(
    infiniopDsv4LinearBf16Fp32Descriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopDsv4LinearBf16Fp32(
    infiniopDsv4LinearBf16Fp32Descriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    const void *w,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyDsv4LinearBf16Fp32Descriptor(
    infiniopDsv4LinearBf16Fp32Descriptor_t desc);

#endif
