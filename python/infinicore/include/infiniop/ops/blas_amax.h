#ifndef __INFINIOP_BLAS_AMAX_API_H__
#define __INFINIOP_BLAS_AMAX_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopBlasAmaxDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateBlasAmaxDescriptor(infiniopHandle_t handle,
                                                                    infiniopBlasAmaxDescriptor_t *desc_ptr,
                                                                    infiniopTensorDescriptor_t x,
                                                                    infiniopTensorDescriptor_t result);

__INFINI_C __export infiniStatus_t infiniopGetBlasAmaxWorkspaceSize(infiniopBlasAmaxDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopBlasAmax(infiniopBlasAmaxDescriptor_t desc,
                                                    void *workspace,
                                                    size_t workspace_size,
                                                    const void *x,
                                                    void *result,
                                                    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyBlasAmaxDescriptor(infiniopBlasAmaxDescriptor_t desc);

#endif // __INFINIOP_BLAS_AMAX_API_H__
