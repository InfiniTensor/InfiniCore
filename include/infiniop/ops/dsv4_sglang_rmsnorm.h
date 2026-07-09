#ifndef __INFINIOP_DSV4_SGLANG_RMSNORM_H__
#define __INFINIOP_DSV4_SGLANG_RMSNORM_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopDsv4SglangRmsnormDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateDsv4SglangRmsnormDescriptor(infiniopHandle_t handle,
                                                                             infiniopDsv4SglangRmsnormDescriptor_t *desc_ptr,
                                                                             infiniopTensorDescriptor_t output_desc,
                                                                             infiniopTensorDescriptor_t input_desc,
                                                                             double eps);

__INFINI_C __export infiniStatus_t infiniopGetDsv4SglangRmsnormWorkspaceSize(infiniopDsv4SglangRmsnormDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopDsv4SglangRmsnorm(infiniopDsv4SglangRmsnormDescriptor_t desc,
                                                             void *workspace,
                                                             size_t workspace_size,
                                                             void *output,
                                                             const void *input,
                                                             void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyDsv4SglangRmsnormDescriptor(infiniopDsv4SglangRmsnormDescriptor_t desc);

#endif
