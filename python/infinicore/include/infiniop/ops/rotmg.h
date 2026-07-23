#ifndef __INFINIOP_ROTMG_API_H__
#define __INFINIOP_ROTMG_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopRotmgDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateRotmgDescriptor(infiniopHandle_t handle,
                                                                 infiniopRotmgDescriptor_t *desc_ptr,
                                                                 infiniopTensorDescriptor_t d1,
                                                                 infiniopTensorDescriptor_t d2,
                                                                 infiniopTensorDescriptor_t x1,
                                                                 infiniopTensorDescriptor_t y1,
                                                                 infiniopTensorDescriptor_t param);

__INFINI_C __export infiniStatus_t infiniopGetRotmgWorkspaceSize(infiniopRotmgDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopRotmg(infiniopRotmgDescriptor_t desc,
                                                 void *workspace,
                                                 size_t workspace_size,
                                                 void *d1,
                                                 void *d2,
                                                 void *x1,
                                                 const void *y1,
                                                 void *param,
                                                 void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyRotmgDescriptor(infiniopRotmgDescriptor_t desc);

#endif // __INFINIOP_ROTMG_API_H__
