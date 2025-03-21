#ifndef __INFINIOP_QUANTIZE_GPTQ_API_H__
#define __INFINIOP_QUANTIZE_GPTQ_API_H__

#include "../operator_descriptor.h"

typedef InfiniopDescriptor *infiniopQuantizeGPTQDescriptor_t;

__C __export infiniStatus_t infiniopCreateQuantizeGPTQDescriptor(infiniopHandle_t handle,
                                                                 infiniopQuantizeGPTQDescriptor_t *desc_ptr,
                                                                 infiniopTensorDescriptor_t c_desc,
                                                                 infiniopTensorDescriptor_t a_desc,
                                                                 infiniopTensorDescriptor_t packed_weights_desc,
                                                                 infiniopTensorDescriptor_t b_scale_desc,
                                                                 infiniopTensorDescriptor_t zero_desc);

__C __export infiniStatus_t infiniopGetQuantizeGPTQWorkspaceSize(infiniopQuantizeGPTQDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopQuantizeGPTQ(infiniopQuantizeGPTQDescriptor_t desc,
                                                 void *workspace,
                                                 size_t workspace_size,
                                                 void *packed_weights,
                                                 void *b_scale,
                                                 void *zero,
                                                 const void *a,
                                                 const void *b,
                                                 void *stream);

__C __export infiniStatus_t infiniopQuantizeLinearGPTQ(infiniopQuantizeGPTQDescriptor_t desc,
                                                       void *workspace,
                                                       size_t workspace_size,
                                                       void *c,
                                                       const void *a,
                                                       void *packed_weights,
                                                       void *b_scale,
                                                       void *zero,
                                                       void *stream);

__C __export infiniStatus_t infiniopDestroyQuantizeGPTQDescriptor(infiniopQuantizeGPTQDescriptor_t desc);

#endif
