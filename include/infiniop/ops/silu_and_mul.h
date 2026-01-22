#ifndef __INFINIOP_SILU_AND_MUL_API_H__
#define __INFINIOP_SILU_AND_MUL_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopSiluAndMulDescriptor_t;

__C __export infiniStatus_t infiniopCreateSiluAndMulDescriptor(
    infiniopHandle_t handle,
    infiniopSiluAndMulDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output,
    infiniopTensorDescriptor_t input);

__C __export infiniStatus_t infiniopGetSiluAndMulWorkspaceSize(
    infiniopSiluAndMulDescriptor_t desc,
    size_t *size);

__C __export infiniStatus_t infiniopSiluAndMul(
    infiniopSiluAndMulDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream);

__C __export infiniStatus_t infiniopDestroySiluAndMulDescriptor(
    infiniopSiluAndMulDescriptor_t desc);

#endif // __INFINIOP_SILU_AND_MUL_API_H__
