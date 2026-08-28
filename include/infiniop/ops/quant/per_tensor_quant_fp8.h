#ifndef __INFINIOP_PER_TENSOR_QUANT_FP8_API_H__
#define __INFINIOP_PER_TENSOR_QUANT_FP8_API_H__

#include "../../operator_descriptor.h"

typedef InfiniopDescriptor *infiniopPerTensorQuantFp8Descriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreatePerTensorQuantFp8Descriptor(infiniopHandle_t handle,
                                                                             infiniopPerTensorQuantFp8Descriptor_t *desc_ptr,
                                                                             infiniopTensorDescriptor_t x_packed_desc,
                                                                             infiniopTensorDescriptor_t x_scale_desc,
                                                                             infiniopTensorDescriptor_t x_desc);

__INFINI_C __export infiniStatus_t infiniopGetPerTensorQuantFp8WorkspaceSize(infiniopPerTensorQuantFp8Descriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopPerTensorQuantFp8(infiniopPerTensorQuantFp8Descriptor_t desc,
                                                             void *workspace,
                                                             size_t workspace_size,
                                                             void *x_packed,
                                                             void *x_scale,
                                                             const void *x,
                                                             const bool is_static,
                                                             void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyPerTensorQuantFp8Descriptor(infiniopPerTensorQuantFp8Descriptor_t desc);

#endif
