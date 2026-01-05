#ifndef __INFINIOP_MOD_API_H__
#define __INFINIOP_MOD_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopModDescriptor_t;

__C __export infiniStatus_t infiniopCreateModDescriptor(infiniopHandle_t handle,
                                                        infiniopModDescriptor_t *desc_ptr,
                                                        infiniopTensorDescriptor_t c,
                                                        infiniopTensorDescriptor_t a,
                                                        infiniopTensorDescriptor_t b);

__C __export infiniStatus_t infiniopGetModWorkspaceSize(infiniopModDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopMod(infiniopModDescriptor_t desc,
                                        void *workspace,
                                        size_t workspace_size,
                                        void *c,
                                        const void *a,
                                        const void *b,
                                        void *stream);

__C __export infiniStatus_t infiniopDestroyModDescriptor(infiniopModDescriptor_t desc);

#endif
