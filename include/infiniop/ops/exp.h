#ifndef __INFINIOP_EXP_API_H__
#define __INFINIOP_EXP_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopExpDescriptor_t;

__C __export infiniStatus_t infiniopCreateExpDescriptor(infiniopHandle_t handle,
                                                        infiniopExpDescriptor_t *desc_ptr,
                                                        infiniopTensorDescriptor_t output,
                                                        infiniopTensorDescriptor_t input);

__C __export infiniStatus_t infiniopGetExpWorkspaceSize(infiniopExpDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopExp(infiniopExpDescriptor_t desc,
                                        void *workspace,
                                        size_t workspace_size,
                                        void *output,
                                        const void *input,
                                        void *stream);

__C __export infiniStatus_t infiniopDestroyExpDescriptor(infiniopExpDescriptor_t desc);

#endif