#ifndef __INFINIOP_TRIL_API_H__
#define __INFINIOP_TRIL_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopTrilDescriptor_t;

__C __export infiniStatus_t infiniopCreateTrilDescriptor(infiniopHandle_t handle,
                                                          infiniopTrilDescriptor_t *desc_ptr,
                                                          infiniopTensorDescriptor_t input,
                                                          infiniopTensorDescriptor_t output,
                                                          int diagonal);

__C __export infiniStatus_t infiniopGetTrilWorkspaceSize(infiniopTrilDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopTril(infiniopTrilDescriptor_t desc,
                                          void *workspace,
                                          size_t workspace_size,
                                          void *output,
                                          void *input,
                                          void *stream);


__C __export infiniStatus_t infiniopTrilInplace(infiniopTrilDescriptor_t desc,
                                                 void *workspace,
                                                 size_t workspace_size,
                                                 void *input_output,
                                                 void *stream);

__C __export infiniStatus_t infiniopDestroyTrilDescriptor(infiniopTrilDescriptor_t desc);

#endif