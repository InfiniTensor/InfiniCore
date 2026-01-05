#ifndef __INFINIOP_DIV_API_H__
#define __INFINIOP_DIV_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopDivDescriptor_t;

__C __export infiniStatus_t infiniopCreateDivDescriptor(infiniopHandle_t handle,
                                                        infiniopDivDescriptor_t *desc_ptr,
                                                        infiniopTensorDescriptor_t c,
                                                        infiniopTensorDescriptor_t a,
                                                        infiniopTensorDescriptor_t b);

__C __export infiniStatus_t infiniopGetDivWorkspaceSize(infiniopDivDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopDiv(infiniopDivDescriptor_t desc,
                                        void *workspace,
                                        size_t workspace_size,
                                        void *c,
                                        const void *a,
                                        const void *b,
                                        void *stream);

__C __export infiniStatus_t infiniopDestroyDivDescriptor(infiniopDivDescriptor_t desc);

#endif
