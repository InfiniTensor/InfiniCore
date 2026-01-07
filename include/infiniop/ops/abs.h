#ifndef __INFINIOP_ABS_API_H__
#define __INFINIOP_ABS_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopAbsDescriptor_t;

__C __export infiniStatus_t infiniopCreateAbsDescriptor(infiniopHandle_t handle,
                                                        infiniopAbsDescriptor_t *desc_ptr,
                                                        infiniopTensorDescriptor_t y,
                                                        infiniopTensorDescriptor_t x);

__C __export infiniStatus_t infiniopGetAbsWorkspaceSize(infiniopAbsDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopAbs(infiniopAbsDescriptor_t desc,
                                        void *workspace,
                                        size_t workspace_size,
                                        void *y,
                                        const void *x,
                                        void *stream);

__C __export infiniStatus_t infiniopDestroyAbsDescriptor(infiniopAbsDescriptor_t desc);

#endif
