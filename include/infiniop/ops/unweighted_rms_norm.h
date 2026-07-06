#ifndef INFINIOP_UNWEIGHTED_RMS_NORM_H
#define INFINIOP_UNWEIGHTED_RMS_NORM_H

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopUnweightedRMSNormDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateUnweightedRMSNormDescriptor(
    infiniopHandle_t handle,
    infiniopUnweightedRMSNormDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    float epsilon);

__INFINI_C __export infiniStatus_t infiniopGetUnweightedRMSNormWorkspaceSize(
    infiniopUnweightedRMSNormDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopUnweightedRMSNorm(
    infiniopUnweightedRMSNormDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyUnweightedRMSNormDescriptor(
    infiniopUnweightedRMSNormDescriptor_t desc);

#endif
