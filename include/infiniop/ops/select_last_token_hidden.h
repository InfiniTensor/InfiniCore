#ifndef __INFINIOP_SELECT_LAST_TOKEN_HIDDEN_API_H__
#define __INFINIOP_SELECT_LAST_TOKEN_HIDDEN_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopSelectLastTokenHiddenDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateSelectLastTokenHiddenDescriptor(
    infiniopHandle_t handle,
    infiniopSelectLastTokenHiddenDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t hidden_states_desc,
    infiniopTensorDescriptor_t input_offsets_desc);

__INFINI_C __export infiniStatus_t infiniopSelectLastTokenHidden(
    infiniopSelectLastTokenHiddenDescriptor_t desc,
    void *output,
    const void *hidden_states,
    const void *input_offsets,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroySelectLastTokenHiddenDescriptor(
    infiniopSelectLastTokenHiddenDescriptor_t desc);

#endif
