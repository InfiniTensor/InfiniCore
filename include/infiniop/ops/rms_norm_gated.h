#ifndef __INFINIOP_RMS_NORM_GATED_H__
#define __INFINIOP_RMS_NORM_GATED_H__
#include "../operator_descriptor.h"
typedef struct InfiniopDescriptor *infiniopRMSNormGatedDescriptor_t;
__INFINI_C __export infiniStatus_t infiniopCreateRMSNormGatedDescriptor(infiniopHandle_t handle, infiniopRMSNormGatedDescriptor_t *desc_ptr, infiniopTensorDescriptor_t out, infiniopTensorDescriptor_t x, infiniopTensorDescriptor_t gate, infiniopTensorDescriptor_t weight, float epsilon);
__INFINI_C __export infiniStatus_t infiniopGetRMSNormGatedWorkspaceSize(infiniopRMSNormGatedDescriptor_t desc, size_t *size);
__INFINI_C __export infiniStatus_t infiniopRMSNormGated(infiniopRMSNormGatedDescriptor_t desc, void *workspace, size_t workspace_size, void *out, const void *x, const void *gate, const void *weight, void *stream);
__INFINI_C __export infiniStatus_t infiniopDestroyRMSNormGatedDescriptor(infiniopRMSNormGatedDescriptor_t desc);
#endif
