#ifndef __INFINIOP_INTERPOLATE_NEAREST_H__
#define __INFINIOP_INTERPOLATE_NEAREST_H__

#include "../operator_descriptor.h"

__C typedef struct InfiniopDescriptor *infiniopInterpolateNearestDescriptor_t;

__C infiniStatus_t infiniopCreateInterpolateNearestDescriptor(infiniopHandle_t handle,
                                                              infiniopInterpolateNearestDescriptor_t *desc_ptr,
                                                              infiniopTensorDescriptor_t output_desc,
                                                              infiniopTensorDescriptor_t input_desc);

__C infiniStatus_t infiniopGetInterpolateNearestWorkspaceSize(infiniopInterpolateNearestDescriptor_t desc,
                                                              size_t *size);

__C infiniStatus_t infiniopInterpolateNearest(infiniopInterpolateNearestDescriptor_t desc,
                                              void *workspace,
                                              size_t workspace_size,
                                              void *output,
                                              const void *input,
                                              void *stream);

__C infiniStatus_t infiniopDestroyInterpolateNearestDescriptor(infiniopInterpolateNearestDescriptor_t desc);

#endif // __INFINIOP_INTERPOLATE_NEAREST_H__
