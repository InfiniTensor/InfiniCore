#ifndef __INFINIOP_ASUM_API_H__
#define __INFINIOP_ASUM_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopAsumDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateAsumDescriptor(infiniopHandle_t handle,
                                                                infiniopAsumDescriptor_t *desc_ptr,
                                                                infiniopTensorDescriptor_t x,
                                                                infiniopTensorDescriptor_t result);

__INFINI_C __export infiniStatus_t infiniopGetAsumWorkspaceSize(infiniopAsumDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopAsum(infiniopAsumDescriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                const void *x,
                                                void *result,
                                                void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyAsumDescriptor(infiniopAsumDescriptor_t desc);

#endif // __INFINIOP_ASUM_API_H__
