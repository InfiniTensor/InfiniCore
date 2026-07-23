#ifndef __INFINIOP_MOE_FUSED_DENSE_API_H__
#define __INFINIOP_MOE_FUSED_DENSE_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopMoeFusedDenseDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateMoeFusedDenseDescriptor(
    infiniopHandle_t handle,
    infiniopMoeFusedDenseDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t hidden_states_desc,
    infiniopTensorDescriptor_t w13_desc,
    infiniopTensorDescriptor_t w2_desc,
    infiniopTensorDescriptor_t topk_weights_desc,
    infiniopTensorDescriptor_t topk_ids_desc,
    infiniopTensorDescriptor_t sorted_token_ids_desc,
    infiniopTensorDescriptor_t expert_ids_desc,
    infiniopTensorDescriptor_t num_tokens_post_padded_desc);

__INFINI_C __export infiniStatus_t infiniopGetMoeFusedDenseWorkspaceSize(
    infiniopMoeFusedDenseDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopMoeFusedDense(
    infiniopMoeFusedDenseDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *hidden_states,
    const void *w13,
    const void *w2,
    const void *topk_weights,
    const void *topk_ids,
    const void *sorted_token_ids,
    const void *expert_ids,
    const void *num_tokens_post_padded,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyMoeFusedDenseDescriptor(
    infiniopMoeFusedDenseDescriptor_t desc);

#endif
