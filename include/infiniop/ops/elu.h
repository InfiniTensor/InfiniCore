#ifndef __INFINIOP_ELU_API_H__
#define __INFINIOP_ELU_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopEluDescriptor_t;

__C __export infiniStatus_t infiniopCreateEluDescriptor(infiniopHandle_t handle,
                                                        infiniopEluDescriptor_t *desc_ptr,
                                                        infiniopTensorDescriptor_t output,
                                                        infiniopTensorDescriptor_t input,
                                                        float alpha);

__C __export infiniStatus_t infiniopGetEluWorkspaceSize(infiniopEluDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopElu(infiniopEluDescriptor_t desc,
                                        void *workspace,
                                        size_t workspace_size,
                                        void *output,
                                        const void *input,
                                        void *stream);

__C __export infiniStatus_t infiniopDestroyEluDescriptor(infiniopEluDescriptor_t desc);

#endif // INFINIOP_OPS_ELU_H