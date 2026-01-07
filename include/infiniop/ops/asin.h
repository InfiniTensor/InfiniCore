#ifndef __INFINIOP_ASIN_API_H__
#define __INFINIOP_ASIN_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopAsinDescriptor_t;

__C __export infiniStatus_t infiniopCreateAsinDescriptor(infiniopHandle_t handle,
                                                        infiniopAsinDescriptor_t *desc_ptr,
                                                        infiniopTensorDescriptor_t y,
                                                        infiniopTensorDescriptor_t x);

__C __export infiniStatus_t infiniopGetAsinWorkspaceSize(infiniopAsinDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopAsin(infiniopAsinDescriptor_t desc,
                                        void *workspace,
                                        size_t workspace_size,
                                        void *y,
                                        const void *x,
                                        void *stream);

__C __export infiniStatus_t infiniopDestroyAsinDescriptor(infiniopAsinDescriptor_t desc);

#endif
