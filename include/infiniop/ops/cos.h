#ifndef __INFINIOP_COS_API_H__
#define __INFINIOP_COS_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopCosDescriptor_t;

__C __export infiniStatus_t infiniopCreateCosDescriptor(infiniopHandle_t handle,
                                                        infiniopCosDescriptor_t *desc_ptr,
                                                        infiniopTensorDescriptor_t output,
                                                        infiniopTensorDescriptor_t input);

__C __export infiniStatus_t infiniopGetCosWorkspaceSize(infiniopCosDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopCos(infiniopCosDescriptor_t desc,
                                        void *workspace,
                                        size_t workspace_size,
                                        void *output,
                                        const void *input,
                                        void *stream);

__C __export infiniStatus_t infiniopDestroyCosDescriptor(infiniopCosDescriptor_t desc);

#endif