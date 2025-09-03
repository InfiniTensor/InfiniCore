#ifndef __INFINIOP_DEQUANTIZE_API_H__
#define __INFINIOP_DEQUANTIZE_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopDequantizeDescriptor_t;

__C __export infiniStatus_t infiniopCreateDequantizeDescriptor(infiniopHandle_t handle,
                                                               infiniopDequantizeDescriptor_t *desc_ptr,
                                                               infiniopTensorDescriptor_t out_desc,
                                                               infiniopTensorDescriptor_t qweight_desc,
                                                               infiniopTensorDescriptor_t scales_desc,
                                                               infiniopTensorDescriptor_t zeros_desc);

__C __export infiniStatus_t infiniopGetDequantizeWorkspaceSize(infiniopDequantizeDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopDequantize(infiniopDequantizeDescriptor_t desc,
                                               void *workspace,
                                               size_t workspace_size,
                                               void *out,
                                               const void *qweight,
                                               const void *scales,
                                               const void *zeros,
                                               size_t split_k_iters,
                                               size_t thx,
                                               size_t thy,
                                               void *stream);

__C __export infiniStatus_t infiniopDestroyDequantizeDescriptor(infiniopDequantizeDescriptor_t desc);

#endif
