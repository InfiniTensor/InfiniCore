#ifndef __INFINIOP_MOE_SUM_API_H__
#define __INFINIOP_MOE_SUM_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopMoeSumDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateMoeSumDescriptor(
    infiniopHandle_t handle,
    infiniopMoeSumDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc);

__INFINI_C __export infiniStatus_t infiniopGetMoeSumWorkspaceSize(
    infiniopMoeSumDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopMoeSum(
    infiniopMoeSumDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyMoeSumDescriptor(
    infiniopMoeSumDescriptor_t desc);

#endif
