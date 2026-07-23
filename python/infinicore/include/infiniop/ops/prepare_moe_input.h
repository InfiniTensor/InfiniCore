#ifndef __INFINIOP_PREPARE_MOE_INPUT_API_H__
#define __INFINIOP_PREPARE_MOE_INPUT_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopPrepareMoeInputDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreatePrepareMoeInputDescriptor(
    infiniopHandle_t handle,
    infiniopPrepareMoeInputDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t expert_offsets_desc,
    infiniopTensorDescriptor_t blockscale_offsets_desc,
    infiniopTensorDescriptor_t problem_sizes1_desc,
    infiniopTensorDescriptor_t problem_sizes2_desc,
    infiniopTensorDescriptor_t input_permutation_desc,
    infiniopTensorDescriptor_t output_permutation_desc,
    infiniopTensorDescriptor_t topk_ids_desc,
    size_t num_experts,
    size_t n,
    size_t k);

__INFINI_C __export infiniStatus_t infiniopGetPrepareMoeInputWorkspaceSize(
    infiniopPrepareMoeInputDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopPrepareMoeInput(
    infiniopPrepareMoeInputDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *expert_offsets,
    void *blockscale_offsets,
    void *problem_sizes1,
    void *problem_sizes2,
    void *input_permutation,
    void *output_permutation,
    const void *topk_ids,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyPrepareMoeInputDescriptor(
    infiniopPrepareMoeInputDescriptor_t desc);

#endif
