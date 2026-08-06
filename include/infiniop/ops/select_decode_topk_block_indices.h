#ifndef __INFINIOP_SELECT_DECODE_TOPK_BLOCK_INDICES_API_H__
#define __INFINIOP_SELECT_DECODE_TOPK_BLOCK_INDICES_API_H__

#include "../operator_descriptor.h"

__INFINI_C __export infiniStatus_t infiniopSelectDecodeTopkBlockIndices(
    infiniopHandle_t handle,
    infiniopTensorDescriptor_t topk_indices_desc,
    infiniopTensorDescriptor_t logits_desc,
    infiniopTensorDescriptor_t seq_lens_desc,
    void *topk_indices,
    const void *logits,
    const void *seq_lens,
    void *stream);

#endif
