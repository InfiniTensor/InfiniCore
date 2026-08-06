#ifndef __INFINIOP_FUSED_ROTARY_EMBEDDING_API_H__
#define __INFINIOP_FUSED_ROTARY_EMBEDDING_API_H__

#include "../operator_descriptor.h"

__INFINI_C __export infiniStatus_t infiniopFusedRotaryEmbedding(
    infiniopHandle_t handle,
    infiniopTensorDescriptor_t query_desc,
    infiniopTensorDescriptor_t key_desc,
    infiniopTensorDescriptor_t positions_desc,
    infiniopTensorDescriptor_t cos_sin_cache_desc,
    void *query,
    void *key,
    const void *positions,
    const void *cos_sin_cache,
    int64_t head_size,
    bool is_neox,
    void *stream);

#endif
