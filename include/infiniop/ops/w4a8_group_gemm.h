#ifndef __INFINIOP_W4A8_GROUP_GEMM_API_H__
#define __INFINIOP_W4A8_GROUP_GEMM_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopW4A8GroupGemmDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateW4A8GroupGemmDescriptor(
    infiniopHandle_t handle,
    infiniopW4A8GroupGemmDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t input_scale_desc,
    infiniopTensorDescriptor_t weight_scale_desc,
    infiniopTensorDescriptor_t tokens_per_experts_desc,
    infiniopTensorDescriptor_t sorted_token_ids_desc,
    infiniopTensorDescriptor_t bias_desc,
    bool trans_weight);

__INFINI_C __export infiniStatus_t infiniopW4A8GroupGemm(
    infiniopW4A8GroupGemmDescriptor_t desc,
    void *out,
    const void *input,
    const void *weight,
    const void *input_scale,
    const void *weight_scale,
    const void *tokens_per_experts,
    const void *sorted_token_ids,
    const void *bias,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyW4A8GroupGemmDescriptor(
    infiniopW4A8GroupGemmDescriptor_t desc);

#endif
