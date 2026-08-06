#ifndef __INFINIOP_SELECT_PREFILL_TOPK_BLOCK_INDICES_API_H__
#define __INFINIOP_SELECT_PREFILL_TOPK_BLOCK_INDICES_API_H__

#include "../operator_descriptor.h"

__INFINI_C __export infiniStatus_t infiniopSelectPrefillTopkBlockIndices(
    infiniopHandle_t handle,
    infiniopTensorDescriptor_t topk_indices_desc,
    infiniopTensorDescriptor_t logits_desc,
    infiniopTensorDescriptor_t cu_seqlen_ks_desc,
    infiniopTensorDescriptor_t cu_seqlen_ke_desc,
    void *topk_indices,
    const void *logits,
    const void *cu_seqlen_ks,
    const void *cu_seqlen_ke,
    void *stream);

#endif
