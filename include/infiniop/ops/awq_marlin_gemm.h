#ifndef __INFINIOP_AWQ_MARLIN_GEMM_API_H__
#define __INFINIOP_AWQ_MARLIN_GEMM_API_H__

#include "../operator_descriptor.h"
#include <cstdint>

typedef struct InfiniopDescriptor *infiniopAwqMarlinGemmDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateAwqMarlinGemmDescriptor(infiniopHandle_t handle,
                                                                         infiniopAwqMarlinGemmDescriptor_t *desc_ptr,
                                                                         infiniopTensorDescriptor_t out_desc,
                                                                         infiniopTensorDescriptor_t a_desc,
                                                                         infiniopTensorDescriptor_t b_desc,
                                                                         infiniopTensorDescriptor_t b_bias_desc,
                                                                         infiniopTensorDescriptor_t b_scales_desc,
                                                                         infiniopTensorDescriptor_t a_scales_desc,
                                                                         infiniopTensorDescriptor_t global_scales_desc,
                                                                         infiniopTensorDescriptor_t b_zeros_desc,
                                                                         infiniopTensorDescriptor_t g_idx_desc,
                                                                         infiniopTensorDescriptor_t perm_desc);

__INFINI_C __export infiniStatus_t infiniopGetAwqMarlinGemmWorkspaceSize(infiniopAwqMarlinGemmDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopAwqMarlinGemm(infiniopAwqMarlinGemmDescriptor_t desc,
                                                         void *workspace,
                                                         size_t workspace_size,
                                                         void *c,
                                                         const void *a,
                                                         const void *b,
                                                         void *b_bias,
                                                         void *b_scales,
                                                         void *a_scales,
                                                         void *global_scales,
                                                         void *b_zeros,
                                                         void *g_idx,
                                                         void *perm,
                                                         int64_t b_q_type_id,
                                                         bool is_k_full,
                                                         bool use_atomic_add,
                                                         bool use_fp32_reduce,
                                                         bool is_zp_float,
                                                         void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyAwqMarlinGemmDescriptor(infiniopAwqMarlinGemmDescriptor_t desc);

#endif
