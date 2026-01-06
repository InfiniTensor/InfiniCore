#ifndef __INFINIOP_RANDOM_SAMPLE_BATCHED_API_H__
#define __INFINIOP_RANDOM_SAMPLE_BATCHED_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopRandomSampleBatchedDescriptor_t;

__C __export infiniStatus_t infiniopCreateRandomSampleBatchedDescriptor(
    infiniopHandle_t handle,
    infiniopRandomSampleBatchedDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t result,
    infiniopTensorDescriptor_t probs);

__C __export infiniStatus_t infiniopGetRandomSampleBatchedWorkspaceSize(
    infiniopRandomSampleBatchedDescriptor_t desc,
    size_t *size);

__C __export infiniStatus_t infiniopRandomSampleBatched(
    infiniopRandomSampleBatchedDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *result,
    const void *probs,
    const float *random_val,
    const float *topp,
    const int *topk,
    const float *temperature,
    int batch_size,
    void *stream);

__C __export infiniStatus_t infiniopDestroyRandomSampleBatchedDescriptor(
    infiniopRandomSampleBatchedDescriptor_t desc);

#endif
