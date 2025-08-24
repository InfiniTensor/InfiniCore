#ifndef __INFINIOP_SCATTER_API_H__
#define __INFINIOP_SCATTER_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopScatterDescriptor_t;

__C __export infiniStatus_t infiniopCreateScatterDescriptor(infiniopHandle_t handle,
                                                            infiniopScatterDescriptor_t *desc_ptr,
                                                            infiniopTensorDescriptor_t input,
                                                            infiniopTensorDescriptor_t output,
                                                            infiniopTensorDescriptor_t index,
                                                            infiniopTensorDescriptor_t src,
                                                            int dim);

__C __export infiniStatus_t infiniopGetScatterWorkspaceSize(infiniopScatterDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopScatter(infiniopScatterDescriptor_t desc,
                                            void *workspace,
                                            size_t workspace_size,
                                            void *output,
                                            const void *input,
                                            const void *index,
                                            const void *src,
                                            void *stream);

__C __export infiniStatus_t infiniopDestroyScatterDescriptor(infiniopScatterDescriptor_t desc);

#endif