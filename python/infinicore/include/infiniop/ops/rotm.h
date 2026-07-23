#ifndef __INFINIOP_ROTM_API_H__
#define __INFINIOP_ROTM_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopRotmDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateRotmDescriptor(infiniopHandle_t handle,
                                                                infiniopRotmDescriptor_t *desc_ptr,
                                                                infiniopTensorDescriptor_t x,
                                                                infiniopTensorDescriptor_t y,
                                                                infiniopTensorDescriptor_t param);

__INFINI_C __export infiniStatus_t infiniopGetRotmWorkspaceSize(infiniopRotmDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopRotm(infiniopRotmDescriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                void *x,
                                                void *y,
                                                const void *param,
                                                void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyRotmDescriptor(infiniopRotmDescriptor_t desc);

#endif // __INFINIOP_ROTM_API_H__
