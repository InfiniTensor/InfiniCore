#ifndef __INFINIOP_ADD_RMS_NORM_API_H__
#define __INFINIOP_ADD_RMS_NORM_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopAddRMSNormDescriptor_t;

__C __export infiniStatus_t infiniopCreateAddRMSNormDescriptor(
    infiniopHandle_t handle,
    infiniopAddRMSNormDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x1_desc,
    infiniopTensorDescriptor_t x2_desc,
    infiniopTensorDescriptor_t w_desc,
    float epsilon);

__C __export infiniStatus_t infiniopGetAddRMSNormWorkspaceSize(infiniopAddRMSNormDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopAddRMSNorm(infiniopAddRMSNormDescriptor_t desc, void *workspace, size_t workspace_size,
                                               void *y, const void *x1, const void *x2, const void *w, void *stream);

__C __export infiniStatus_t infiniopDestroyAddRMSNormDescriptor(infiniopAddRMSNormDescriptor_t desc);

#endif
