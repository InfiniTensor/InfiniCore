#ifndef __INFINIOP_BLAS_COPY_API_H__
#define __INFINIOP_BLAS_COPY_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopBlasCopyDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateBlasCopyDescriptor(infiniopHandle_t handle,
                                                                    infiniopBlasCopyDescriptor_t *desc_ptr,
                                                                    infiniopTensorDescriptor_t x,
                                                                    infiniopTensorDescriptor_t y);

__INFINI_C __export infiniStatus_t infiniopGetBlasCopyWorkspaceSize(infiniopBlasCopyDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopBlasCopy(infiniopBlasCopyDescriptor_t desc,
                                                    void *workspace,
                                                    size_t workspace_size,
                                                    const void *x,
                                                    void *y,
                                                    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyBlasCopyDescriptor(infiniopBlasCopyDescriptor_t desc);

#endif // __INFINIOP_BLAS_COPY_API_H__
