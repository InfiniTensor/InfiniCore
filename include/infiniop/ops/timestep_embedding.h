#ifndef __INFINIOP_TIMESTEP_EMBEDDING_API_H__
#define __INFINIOP_TIMESTEP_EMBEDDING_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopTimestepEmbeddingDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateTimestepEmbeddingDescriptor(
    infiniopHandle_t handle,
    infiniopTimestepEmbeddingDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t timestep_desc);

__INFINI_C __export infiniStatus_t infiniopTimestepEmbedding(
    infiniopTimestepEmbeddingDescriptor_t desc,
    void *output,
    const void *timestep,
    float max_period,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyTimestepEmbeddingDescriptor(
    infiniopTimestepEmbeddingDescriptor_t desc);

#endif
