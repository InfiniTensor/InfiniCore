#pragma once
#include "../operator_descriptor.h"
typedef struct InfiniopDescriptor *infiniopGDNQKVNormDescriptor_t;
__INFINI_C __export infiniStatus_t infiniopCreateGDNQKVNormDescriptor(infiniopHandle_t handle, infiniopGDNQKVNormDescriptor_t *desc_ptr, infiniopTensorDescriptor_t q_out, infiniopTensorDescriptor_t k_out, infiniopTensorDescriptor_t v_out, infiniopTensorDescriptor_t q, infiniopTensorDescriptor_t k, infiniopTensorDescriptor_t v);
__INFINI_C __export infiniStatus_t infiniopGetGDNQKVNormWorkspaceSize(infiniopGDNQKVNormDescriptor_t desc, size_t *size);
__INFINI_C __export infiniStatus_t infiniopGDNQKVNorm(infiniopGDNQKVNormDescriptor_t desc, void *workspace, size_t workspace_size, void *q_out, void *k_out, void *v_out, const void *q, const void *k, const void *v, void *stream);
__INFINI_C __export infiniStatus_t infiniopDestroyGDNQKVNormDescriptor(infiniopGDNQKVNormDescriptor_t desc);
