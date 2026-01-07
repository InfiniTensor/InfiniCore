#ifndef __INFINIOP_ATAN_API_H__
#define __INFINIOP_ATAN_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopAtanDescriptor_t;

__C __export infiniStatus_t infiniopCreateAtanDescriptor(infiniopHandle_t handle,
                                                        infiniopAtanDescriptor_t *desc_ptr,
                                                        infiniopTensorDescriptor_t y,
                                                        infiniopTensorDescriptor_t x);

__C __export infiniStatus_t infiniopGetAtanWorkspaceSize(infiniopAtanDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopAtan(infiniopAtanDescriptor_t desc,
                                        void *workspace,
                                        size_t workspace_size,
                                        void *y,
                                        const void *x,
                                        void *stream);

__C __export infiniStatus_t infiniopDestroyAtanDescriptor(infiniopAtanDescriptor_t desc);

#endif
