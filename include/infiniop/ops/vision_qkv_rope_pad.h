#pragma once
#include "../operator_descriptor.h"
typedef struct InfiniopDescriptor *infiniopVisionQKVRoPEPadDescriptor_t;
__INFINI_C __export infiniStatus_t infiniopCreateVisionQKVRoPEPadDescriptor(infiniopHandle_t handle, infiniopVisionQKVRoPEPadDescriptor_t *desc_ptr, infiniopTensorDescriptor_t q_out, infiniopTensorDescriptor_t k_out, infiniopTensorDescriptor_t v_out, infiniopTensorDescriptor_t qkv, infiniopTensorDescriptor_t cos, infiniopTensorDescriptor_t sin, infiniopTensorDescriptor_t rows, infiniopTensorDescriptor_t cols);
__INFINI_C __export infiniStatus_t infiniopGetVisionQKVRoPEPadWorkspaceSize(infiniopVisionQKVRoPEPadDescriptor_t desc, size_t *size);
__INFINI_C __export infiniStatus_t infiniopVisionQKVRoPEPad(infiniopVisionQKVRoPEPadDescriptor_t desc, void *workspace, size_t workspace_size, void *q_out, void *k_out, void *v_out, const void *qkv, const void *cos, const void *sin, const void *rows, const void *cols, void *stream);
__INFINI_C __export infiniStatus_t infiniopDestroyVisionQKVRoPEPadDescriptor(infiniopVisionQKVRoPEPadDescriptor_t desc);
