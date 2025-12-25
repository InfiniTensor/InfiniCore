#ifndef __INFINIOP_BI_ATTENTION_API_H__
#define __INFINIOP_BI_ATTENTION_API_H__

#include "../operator_descriptor.h"
#include "gemm.h"
#include "swiglu.h"

typedef struct InfiniopDescriptor *infiniopBiAttentionDescriptor_t;

__C __export infiniStatus_t infiniopCreateBiAttentionDescriptor(infiniopHandle_t handle,
                                                              infiniopBiAttentionDescriptor_t *desc_ptr,
                                                              infiniopTensorDescriptor_t out_desc,
                                                              infiniopTensorDescriptor_t q_desc,
                                                              infiniopTensorDescriptor_t k_desc,
                                                              infiniopTensorDescriptor_t v_desc,
                                                              infiniopTensorDescriptor_t k_cache_desc,
                                                              infiniopTensorDescriptor_t v_cache_desc,
                                                              size_t pos);

__C __export infiniStatus_t infiniopGetBiAttentionWorkspaceSize(infiniopBiAttentionDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopBiAttention(infiniopBiAttentionDescriptor_t desc,
                                              void *workspace,
                                              size_t workspace_size,
                                              void *out,
                                              const void *q,
                                              const void *k,
                                              const void *v,
                                              void *k_cache,
                                              void *v_cache,
                                              void *stream);

__C __export infiniStatus_t infiniopDestroyBiAttentionDescriptor(infiniopBiAttentionDescriptor_t desc);
#endif
