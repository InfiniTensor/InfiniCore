#ifndef __INFINIOP_BLAS_DOT_API_H__
#define __INFINIOP_BLAS_DOT_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopBlasDotDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateBlasDotDescriptor(infiniopHandle_t handle,
                                                                   infiniopBlasDotDescriptor_t *desc_ptr,
                                                                   infiniopTensorDescriptor_t x,
                                                                   infiniopTensorDescriptor_t y,
                                                                   infiniopTensorDescriptor_t result);

__INFINI_C __export infiniStatus_t infiniopGetBlasDotWorkspaceSize(infiniopBlasDotDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopBlasDot(infiniopBlasDotDescriptor_t desc,
                                                   void *workspace,
                                                   size_t workspace_size,
                                                   const void *x,
                                                   const void *y,
                                                   void *result,
                                                   void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyBlasDotDescriptor(infiniopBlasDotDescriptor_t desc);

#endif // __INFINIOP_BLAS_DOT_API_H__
