#ifndef __INFINIOP_AXPY_API_H__
#define __INFINIOP_AXPY_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopAxpyDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateAxpyDescriptor(infiniopHandle_t handle,
                                                                infiniopAxpyDescriptor_t *desc_ptr,
                                                                infiniopTensorDescriptor_t alpha,
                                                                infiniopTensorDescriptor_t x,
                                                                infiniopTensorDescriptor_t y);

__INFINI_C __export infiniStatus_t infiniopGetAxpyWorkspaceSize(infiniopAxpyDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopAxpy(infiniopAxpyDescriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                const void *alpha,
                                                const void *x,
                                                void *y,
                                                void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyAxpyDescriptor(infiniopAxpyDescriptor_t desc);

#endif // __INFINIOP_AXPY_API_H__
