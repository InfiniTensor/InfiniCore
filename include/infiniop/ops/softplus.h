#ifndef __INFINIOP_OPS_SOFTPLUS_H__
#define __INFINIOP_OPS_SOFTPLUS_H__
#include "../tensor_descriptor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct InfiniopSoftplusDescriptor *infiniopSoftplusDescriptor_t;
__C __export infiniStatus_t infiniopCreateSoftplusDescriptor(
    infiniopHandle_t handle,
    infiniopSoftplusDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    float beta,       
    float threshold   
);

__C __export infiniStatus_t infiniopGetSoftplusWorkspaceSize(
    infiniopSoftplusDescriptor_t desc, 
    size_t *size);

__C __export infiniStatus_t infiniopSoftplus(
    infiniopSoftplusDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void *stream);

__C __export infiniStatus_t infiniopDestroySoftplusDescriptor(
    infiniopSoftplusDescriptor_t desc);

#ifdef __cplusplus
}
#endif

#endif // __INFINIOP_OPS_SOFTPLUS_H__