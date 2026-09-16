#ifndef __INFINIOP_W4A8_MOE_SHUFFLE_API_H__
#define __INFINIOP_W4A8_MOE_SHUFFLE_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopW4A8MoeShuffleDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateW4A8MoeShuffleDescriptor(
    infiniopHandle_t handle,
    infiniopW4A8MoeShuffleDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc);

__INFINI_C __export infiniStatus_t infiniopW4A8MoeShuffle(
    infiniopW4A8MoeShuffleDescriptor_t desc,
    void *output,
    const void *input,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyW4A8MoeShuffleDescriptor(
    infiniopW4A8MoeShuffleDescriptor_t desc);

#endif
