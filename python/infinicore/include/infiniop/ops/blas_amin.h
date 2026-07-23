#ifndef __INFINIOP_BLAS_AMIN_API_H__
#define __INFINIOP_BLAS_AMIN_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopBlasAminDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateBlasAminDescriptor(infiniopHandle_t handle,
                                                                    infiniopBlasAminDescriptor_t *desc_ptr,
                                                                    infiniopTensorDescriptor_t x,
                                                                    infiniopTensorDescriptor_t result);

__INFINI_C __export infiniStatus_t infiniopGetBlasAminWorkspaceSize(infiniopBlasAminDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopBlasAmin(infiniopBlasAminDescriptor_t desc,
                                                    void *workspace,
                                                    size_t workspace_size,
                                                    const void *x,
                                                    void *result,
                                                    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyBlasAminDescriptor(infiniopBlasAminDescriptor_t desc);

#endif // __INFINIOP_BLAS_AMIN_API_H__
