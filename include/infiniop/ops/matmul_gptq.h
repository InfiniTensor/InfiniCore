#ifndef __INFINIOP_MATMUL_GPTQ_API_H__
#define __INFINIOP_MATMUL_GPTQ_API_H__

#include "../operator_descriptor.h"

typedef InfiniopDescriptor *infiniopMatmulGptqDescriptor_t;

__C __export infiniStatus_t infiniopCreateMatmulGptqDescriptor(infiniopHandle_t handle,
                                                               infiniopMatmulGptqDescriptor_t *desc_ptr,
                                                               infiniopTensorDescriptor_t c_desc,
                                                               infiniopTensorDescriptor_t a_desc,
                                                               infiniopTensorDescriptor_t packed_weights_desc,
                                                               infiniopTensorDescriptor_t b_scale_desc,
                                                               infiniopTensorDescriptor_t zero_desc);

__C __export infiniStatus_t infiniopGetMatmulGptqWorkspaceSize(infiniopMatmulGptqDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopMatmulQuant(infiniopMatmulGptqDescriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                void *packed_weights,
                                                void *b_scale,
                                                void *zero,
                                                const void *a,
                                                const void *b,
                                                void *stream);

__C __export infiniStatus_t infiniopMatmulGptq(infiniopMatmulGptqDescriptor_t desc,
                                               void *workspace,
                                               size_t workspace_size,
                                               void *c,
                                               const void *a,
                                               void *packed_weights,
                                               void *b_scale,
                                               void *zero,
                                               void *stream);

__C __export infiniStatus_t infiniopDestroyMatmulGptqDescriptor(infiniopMatmulGptqDescriptor_t desc);

#endif
