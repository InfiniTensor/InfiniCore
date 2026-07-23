#ifndef __INFINIOP_SCAL_API_H__
#define __INFINIOP_SCAL_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopScalDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateScalDescriptor(infiniopHandle_t handle,
                                                                infiniopScalDescriptor_t *desc_ptr,
                                                                infiniopTensorDescriptor_t alpha,
                                                                infiniopTensorDescriptor_t x);

__INFINI_C __export infiniStatus_t infiniopGetScalWorkspaceSize(infiniopScalDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopScal(infiniopScalDescriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                const void *alpha,
                                                void *x,
                                                void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyScalDescriptor(infiniopScalDescriptor_t desc);

#endif // __INFINIOP_SCAL_API_H__
