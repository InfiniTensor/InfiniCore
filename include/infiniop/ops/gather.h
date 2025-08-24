#ifndef __INFINIOP_GATHER_API_H__
#define __INFINIOP_GATHER_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopGatherDescriptor_t;

__C __export infiniStatus_t infiniopCreateGatherDescriptor(infiniopHandle_t handle,
                                                            infiniopGatherDescriptor_t *desc_ptr,
                                                            infiniopTensorDescriptor_t input,
                                                            infiniopTensorDescriptor_t output,
                                                            int dim,
                                                            infiniopTensorDescriptor_t index);

__C __export infiniStatus_t infiniopGetGatherWorkspaceSize(infiniopGatherDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopGather(infiniopGatherDescriptor_t desc,
                                            void *workspace,
                                            size_t workspace_size,
                                            void *output,
                                            void *input,
                                            const void *index,
                                            void *stream);

__C __export infiniStatus_t infiniopDestroyGatherDescriptor(infiniopGatherDescriptor_t desc);

#endif