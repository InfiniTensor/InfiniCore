#pragma once
#include "../operator_descriptor.h"
typedef struct InfiniopDescriptor *infiniopQKRMSNormRoPEDescriptor_t;
__INFINI_C __export infiniStatus_t infiniopCreateQKRMSNormRoPEDescriptor(
    infiniopHandle_t handle, infiniopQKRMSNormRoPEDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t q_out,
    infiniopTensorDescriptor_t k_out,
    infiniopTensorDescriptor_t q,
    infiniopTensorDescriptor_t k,
    infiniopTensorDescriptor_t q_weight,
    infiniopTensorDescriptor_t k_weight,
    infiniopTensorDescriptor_t cos,
    infiniopTensorDescriptor_t sin,
    infiniopTensorDescriptor_t positions, float epsilon);
__INFINI_C __export infiniStatus_t infiniopGetQKRMSNormRoPEWorkspaceSize(infiniopQKRMSNormRoPEDescriptor_t desc, size_t *size);
__INFINI_C __export infiniStatus_t infiniopQKRMSNormRoPE(
    infiniopQKRMSNormRoPEDescriptor_t desc, void *workspace, size_t workspace_size,
    void *q_out,
    void *k_out,
    const void *q,
    const void *k,
    const void *q_weight,
    const void *k_weight,
    const void *cos,
    const void *sin,
    const void *positions, void *stream);
__INFINI_C __export infiniStatus_t infiniopDestroyQKRMSNormRoPEDescriptor(infiniopQKRMSNormRoPEDescriptor_t desc);
