#ifndef __INFINIOP_MUL_SCALAR_API_H__
#define __INFINIOP_MUL_SCALAR_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopMulScalarDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateMulScalarDescriptor(infiniopHandle_t handle,
                                                                     infiniopMulScalarDescriptor_t *desc_ptr,
                                                                     infiniopTensorDescriptor_t output,
                                                                     infiniopTensorDescriptor_t input);

__INFINI_C __export infiniStatus_t infiniopGetMulScalarWorkspaceSize(infiniopMulScalarDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopMulScalar(infiniopMulScalarDescriptor_t desc,
                                                     void *workspace,
                                                     size_t workspace_size,
                                                     void *output,
                                                     const void *input,
                                                     double alpha,
                                                     void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyMulScalarDescriptor(infiniopMulScalarDescriptor_t desc);

#endif
