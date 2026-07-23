#ifndef __INFINIOP_NRM2_API_H__
#define __INFINIOP_NRM2_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopNrm2Descriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateNrm2Descriptor(infiniopHandle_t handle,
                                                                infiniopNrm2Descriptor_t *desc_ptr,
                                                                infiniopTensorDescriptor_t x,
                                                                infiniopTensorDescriptor_t result);

__INFINI_C __export infiniStatus_t infiniopGetNrm2WorkspaceSize(infiniopNrm2Descriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopNrm2(infiniopNrm2Descriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                const void *x,
                                                void *result,
                                                void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyNrm2Descriptor(infiniopNrm2Descriptor_t desc);

#endif // __INFINIOP_NRM2_API_H__
