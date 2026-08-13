#ifndef __INFINIOP_MATMUL_ALL_REDUCE_API_H__
#define __INFINIOP_MATMUL_ALL_REDUCE_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopMatmulAllReduceDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateMatmulAllReduceDescriptor(
    infiniopHandle_t handle,
    infiniopMatmulAllReduceDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t bias_desc,
    const char *group_name);

__INFINI_C __export infiniStatus_t infiniopGetMatmulAllReduceWorkspaceSize(
    infiniopMatmulAllReduceDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopMatmulAllReduce(
    infiniopMatmulAllReduceDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    const void *weight,
    const void *bias,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyMatmulAllReduceDescriptor(
    infiniopMatmulAllReduceDescriptor_t desc);

#endif
