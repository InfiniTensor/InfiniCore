#ifndef __INFINIOP_CAST_API_H__
#define __INFINIOP_CAST_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopCastDescriptor_t;

__C __export infiniStatus_t infiniopCreateCastDescriptor(infiniopHandle_t handle,
                                                         infiniopCastDescriptor_t *desc_ptr,
                                                         infiniopTensorDescriptor_t output,
                                                         infiniopTensorDescriptor_t input);

__C __export infiniStatus_t infiniopGetCastWorkspaceSize(infiniopCastDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopCast(infiniopCastDescriptor_t desc,
                                         void *workspace,
                                         size_t workspace_size,
                                         void *output,
                                         const void *input,
                                         void *stream);

__C __export infiniStatus_t infiniopDestroyCastDescriptor(infiniopCastDescriptor_t desc);

#endif