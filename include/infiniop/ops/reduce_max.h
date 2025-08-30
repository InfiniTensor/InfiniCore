#ifndef __INFINIOP_REDUCE_MAX_API_H__
#define __INFINIOP_REDUCE_MAX_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopReduceMaxDescriptor_t;

__C __export infiniStatus_t infiniopCreateReduceMaxDescriptor(
    infiniopHandle_t handle,
    infiniopReduceMaxDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    size_t dim);

__C __export infiniStatus_t infiniopGetReduceMaxWorkspaceSize(infiniopReduceMaxDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopReduceMax(
    infiniopReduceMaxDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream);

__C __export infiniStatus_t infiniopDestroyReduceMaxDescriptor(infiniopReduceMaxDescriptor_t desc);

#endif
