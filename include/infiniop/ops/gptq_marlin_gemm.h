#ifndef __INFINIOP_GPTQ_MARLIN_GEMM_API_H__
#define __INFINIOP_GPTQ_MARLIN_GEMM_API_H__

#include "../operator_descriptor.h"
#include <cstdint>

typedef struct InfiniopDescriptor *infiniopGptqMarlinGemmDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateGptqMarlinGemmDescriptor(infiniopHandle_t handle,
                                                                          infiniopGptqMarlinGemmDescriptor_t *desc_ptr,
                                                                          infiniopTensorDescriptor_t out_desc,
                                                                          infiniopTensorDescriptor_t a_desc,
                                                                          infiniopTensorDescriptor_t b_desc,
                                                                          infiniopTensorDescriptor_t b_scales_desc,
                                                                          infiniopTensorDescriptor_t global_scale_desc,
                                                                          infiniopTensorDescriptor_t b_zeros_desc,
                                                                          infiniopTensorDescriptor_t g_idx_desc,
                                                                          infiniopTensorDescriptor_t perm_desc);

__INFINI_C __export infiniStatus_t infiniopGetGptqMarlinGemmWorkspaceSize(infiniopGptqMarlinGemmDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopGptqMarlinGemm(infiniopGptqMarlinGemmDescriptor_t desc,
                                                          void *workspace,
                                                          size_t workspace_size,
                                                          void *out,
                                                          const void *a,
                                                          const void *b,
                                                          void *b_scales,
                                                          void *global_scale,
                                                          void *b_zeros,
                                                          void *g_idx,
                                                          void *perm,
                                                          int64_t b_q_type_id,
                                                          bool is_k_full,
                                                          bool use_atomic_add,
                                                          bool use_fp32_reduce,
                                                          bool is_zp_float,
                                                          void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyGptqMarlinGemmDescriptor(infiniopGptqMarlinGemmDescriptor_t desc);

#endif
