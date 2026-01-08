#ifndef __INFINIOP_AVERAGEPOOL_H__
#define __INFINIOP_AVERAGEPOOL_H__

#include "../operator_descriptor.h"

__C typedef struct InfiniopDescriptor *infiniopAvgPoolDescriptor_t;

__C infiniStatus_t infiniopCreateAvgPoolDescriptor(infiniopHandle_t handle,
                                                   infiniopAvgPoolDescriptor_t *desc_ptr,
                                                   infiniopTensorDescriptor_t output_desc,
                                                   infiniopTensorDescriptor_t input_desc,
                                                   void *kernel_size,
                                                   void *strides,
                                                   void *pads,
                                                   bool ceil_mode);

__C infiniStatus_t infiniopGetAvgPoolWorkspaceSize(infiniopAvgPoolDescriptor_t desc,
                                                   size_t *size);

__C infiniStatus_t infiniopAvgPool(infiniopAvgPoolDescriptor_t desc,
                                   void *workspace,
                                   size_t workspace_size,
                                   void *output,
                                   const void *input,
                                   void *stream);

__C infiniStatus_t infiniopDestroyAvgPoolDescriptor(infiniopAvgPoolDescriptor_t desc);

#endif // __INFINIOP_AVERAGEPOOL_H__
