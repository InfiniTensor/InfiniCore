#ifndef __INFINIOP_MOE_WNA16_MARLIN_GEMM_API_H__
#define __INFINIOP_MOE_WNA16_MARLIN_GEMM_API_H__

#include "../operator_descriptor.h"
#include <cstdint>

typedef struct InfiniopDescriptor *infiniopMoeWna16MarlinGemmDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateMoeWna16MarlinGemmDescriptor(infiniopHandle_t handle,
                                                                              infiniopMoeWna16MarlinGemmDescriptor_t *desc_ptr,
                                                                              infiniopTensorDescriptor_t c_desc,
                                                                              infiniopTensorDescriptor_t a_desc,
                                                                              infiniopTensorDescriptor_t b_q_weight_desc,
                                                                              infiniopTensorDescriptor_t b_bias_desc,
                                                                              infiniopTensorDescriptor_t b_scales_desc,
                                                                              infiniopTensorDescriptor_t global_scales_desc,
                                                                              infiniopTensorDescriptor_t b_zeros_desc,
                                                                              infiniopTensorDescriptor_t g_idx_desc,
                                                                              infiniopTensorDescriptor_t perm_desc,
                                                                              infiniopTensorDescriptor_t sorted_token_desc,
                                                                              infiniopTensorDescriptor_t expert_ids_desc,
                                                                              infiniopTensorDescriptor_t num_tokens_post_padded_desc,
                                                                              infiniopTensorDescriptor_t topk_weights_desc, 
                                                                              int size_m, int size_n, int size_k,
                                                                              int top_k, int moe_block_size);
;

__INFINI_C __export infiniStatus_t infiniopGetMoeWna16MarlinGemmWorkspaceSize(infiniopMoeWna16MarlinGemmDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopMoeWna16MarlinGemm(infiniopMoeWna16MarlinGemmDescriptor_t desc,
                                                              void *workspace,
                                                              size_t workspace_size,
                                                              void *c,
                                                              const void *a,
                                                              const void *b_q_weight,
                                                              void *b_bias,
                                                              void *b_scales,
                                                              void *global_scales,
                                                              void *b_zeros,
                                                              void *g_idx,
                                                              void *perm,
                                                              void *sorted_token_ids,
                                                              void *expert_ids,
                                                              void *num_tokens_post_padded,
                                                              void *topk_weights,
                                                              bool mul_topk_weights,
                                                              bool is_ep,
                                                              int64_t b_q_type_id,
                                                              bool is_k_full,
                                                              bool use_atomic_add,
                                                              bool use_fp32_reduce,
                                                              bool is_zp_float,
                                                              void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyMoeWna16MarlinGemmDescriptor(infiniopMoeWna16MarlinGemmDescriptor_t desc);

#endif
