#ifndef __INFINIOP_CONVERT_TO_F32_API_H__
#define __INFINIOP_CONVERT_TO_F32_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopConvertToF32Descriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateConvertToF32Descriptor(infiniopHandle_t handle,
                                                                          infiniopConvertToF32Descriptor_t *desc_ptr,
                                                                          infiniopTensorDescriptor_t y,
                                                                          infiniopTensorDescriptor_t x);

__INFINI_C __export infiniStatus_t infiniopGetConvertToF32WorkspaceSize(infiniopConvertToF32Descriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopConvertToF32(infiniopConvertToF32Descriptor_t desc,
                                                        void *workspace,
                                                        size_t workspace_size,
                                                        void *y,
                                                        const void *x,
                                                        void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyConvertToF32Descriptor(infiniopConvertToF32Descriptor_t desc);

#endif
