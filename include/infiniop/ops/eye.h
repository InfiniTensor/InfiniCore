#ifndef __INFINIOP_EYE_API_H__
#define __INFINIOP_EYE_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopEyeDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateEyeDescriptor(infiniopHandle_t handle,
                                                        infiniopEyeDescriptor_t *desc_ptr,
                                                        infiniopTensorDescriptor_t y);

__INFINI_C __export infiniStatus_t infiniopGetEyeWorkspaceSize(infiniopEyeDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopEye(infiniopEyeDescriptor_t desc,
                                        void *workspace,
                                        size_t workspace_size,
                                        void *y,
                                        void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyEyeDescriptor(infiniopEyeDescriptor_t desc);

#endif
