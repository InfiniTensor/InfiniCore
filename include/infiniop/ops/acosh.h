#ifndef __INFINIOP_ACOSH_API_H__
#define __INFINIOP_ACOSH_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopAcoshDescriptor_t;

__C __export infiniStatus_t infiniopCreateAcoshDescriptor(infiniopHandle_t handle,
                                                        infiniopAcoshDescriptor_t *desc_ptr,
                                                        infiniopTensorDescriptor_t y,
                                                        infiniopTensorDescriptor_t x);

__C __export infiniStatus_t infiniopGetAcoshWorkspaceSize(infiniopAcoshDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopAcosh(infiniopAcoshDescriptor_t desc,
                                        void *workspace,
                                        size_t workspace_size,
                                        void *y,
                                        const void *x,
                                        void *stream);

__C __export infiniStatus_t infiniopDestroyAcoshDescriptor(infiniopAcoshDescriptor_t desc);

#endif
