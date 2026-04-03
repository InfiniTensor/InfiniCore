#ifndef __INFINIOP_GPTQ_GEMM_API_H__
#define __INFINIOP_GPTQ_GEMM_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopGptqGemmDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateGptqGemmDescriptor(
    infiniopHandle_t handle,
    infiniopGptqGemmDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t b_scales_desc,
    infiniopTensorDescriptor_t b_zeros_desc,
    infiniopTensorDescriptor_t b_g_idx_desc,
    bool use_exllama,
    int quant_bit);

__INFINI_C __export infiniStatus_t infiniopGetGptqGemmWorkspaceSize(
    infiniopGptqGemmDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopGptqGemm(
    infiniopGptqGemmDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *a,
    const void *b,
    const void *b_scale,
    const void *b_zero,
    const void *b_g_idx,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyGptqGemmDescriptor(
    infiniopGptqGemmDescriptor_t desc);
#endif
