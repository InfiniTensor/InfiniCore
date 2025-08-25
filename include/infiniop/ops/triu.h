#ifndef __INFINIOP_TRIU_API_H__
#define __INFINIOP_TRIU_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopTriuDescriptor_t;

__C __export infiniStatus_t infiniopCreateTriuDescriptor(infiniopHandle_t handle,
                                                          infiniopTriuDescriptor_t *desc_ptr,
                                                          infiniopTensorDescriptor_t input,
                                                          infiniopTensorDescriptor_t output,
                                                          int diagonal);

__C __export infiniStatus_t infiniopGetTriuWorkspaceSize(infiniopTriuDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopTriu(infiniopTriuDescriptor_t desc,
                                          void *workspace,
                                          size_t workspace_size,
                                          void *output,
                                          void *input,
                                          void *stream);

__C __export infiniStatus_t infiniopTriuInplace(infiniopTriuDescriptor_t desc,
                                                 void *workspace,
                                                 size_t workspace_size,
                                                 void *input_output,
                                                 void *stream);

__C __export infiniStatus_t infiniopDestroyTriuDescriptor(infiniopTriuDescriptor_t desc);

#endif