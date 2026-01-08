#ifndef __INFINIOP_MAX_POOL_H__
#define __INFINIOP_MAX_POOL_H__

#include "../operator_descriptor.h"

__C typedef struct InfiniopDescriptor *infiniopMaxPoolDescriptor_t;

__C infiniStatus_t infiniopCreateMaxPoolDescriptor(infiniopHandle_t handle,
                                                   infiniopMaxPoolDescriptor_t *desc_ptr,
                                                   infiniopTensorDescriptor_t output_desc,
                                                   infiniopTensorDescriptor_t input_desc,
                                                   void *kernel_size,
                                                   void *strides,
                                                   void *pads,
                                                   bool ceil_mode);

__C infiniStatus_t infiniopGetMaxPoolWorkspaceSize(infiniopMaxPoolDescriptor_t desc,
                                                   size_t *size);

__C infiniStatus_t infiniopMaxPool(infiniopMaxPoolDescriptor_t desc,
                                   void *workspace,
                                   size_t workspace_size,
                                   void *output,
                                   const void *input,
                                   void *stream);

__C infiniStatus_t infiniopDestroyMaxPoolDescriptor(infiniopMaxPoolDescriptor_t desc);

#endif // __INFINIOP_MAX_POOL_H__
