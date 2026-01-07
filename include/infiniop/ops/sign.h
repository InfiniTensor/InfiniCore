#ifndef __INFINIOP_SIGN_API_H__
#define __INFINIOP_SIGN_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopSignDescriptor_t;

__C __export infiniStatus_t infiniopCreateSignDescriptor(infiniopHandle_t handle,
                                                        infiniopSignDescriptor_t *desc_ptr,
                                                        infiniopTensorDescriptor_t y,
                                                        infiniopTensorDescriptor_t x);

__C __export infiniStatus_t infiniopGetSignWorkspaceSize(infiniopSignDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopSign(infiniopSignDescriptor_t desc,
                                        void *workspace,
                                        size_t workspace_size,
                                        void *y,
                                        const void *x,
                                        void *stream);

__C __export infiniStatus_t infiniopDestroySignDescriptor(infiniopSignDescriptor_t desc);

#endif
