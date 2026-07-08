#ifndef __INFINIOP_DSV4_PER_TOKEN_QUANT_INT8_API_H__
#define __INFINIOP_DSV4_PER_TOKEN_QUANT_INT8_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopDsv4PerTokenQuantInt8Descriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateDsv4PerTokenQuantInt8Descriptor(
    infiniopHandle_t handle,
    infiniopDsv4PerTokenQuantInt8Descriptor_t *desc_ptr,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t scale_desc,
    infiniopTensorDescriptor_t x_desc);

__INFINI_C __export infiniStatus_t infiniopGetDsv4PerTokenQuantInt8WorkspaceSize(
    infiniopDsv4PerTokenQuantInt8Descriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopDsv4PerTokenQuantInt8(
    infiniopDsv4PerTokenQuantInt8Descriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *q,
    void *scale,
    const void *x,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyDsv4PerTokenQuantInt8Descriptor(
    infiniopDsv4PerTokenQuantInt8Descriptor_t desc);

#endif // __INFINIOP_DSV4_PER_TOKEN_QUANT_INT8_API_H__
