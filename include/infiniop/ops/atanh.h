#ifndef __INFINIOP_ATANH_API_H__
#define __INFINIOP_ATANH_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopAtanhDescriptor_t;

__C __export infiniStatus_t infiniopCreateAtanhDescriptor(infiniopHandle_t handle,
                                                        infiniopAtanhDescriptor_t *desc_ptr,
                                                        infiniopTensorDescriptor_t y,
                                                        infiniopTensorDescriptor_t x);

__C __export infiniStatus_t infiniopGetAtanhWorkspaceSize(infiniopAtanhDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopAtanh(infiniopAtanhDescriptor_t desc,
                                        void *workspace,
                                        size_t workspace_size,
                                        void *y,
                                        const void *x,
                                        void *stream);

__C __export infiniStatus_t infiniopDestroyAtanhDescriptor(infiniopAtanhDescriptor_t desc);

#endif
