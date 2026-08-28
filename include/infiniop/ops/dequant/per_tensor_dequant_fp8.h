#ifndef __INFINIOP_PER_TENSOR_DEQUANT_FP8_API_H__
#define __INFINIOP_PER_TENSOR_DEQUANT_FP8_API_H__

#include "../../operator_descriptor.h"

typedef InfiniopDescriptor *infiniopPerTensorDequantFp8Descriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreatePerTensorDequantFp8Descriptor(infiniopHandle_t handle,
                                                                               infiniopPerTensorDequantFp8Descriptor_t *desc_ptr,
                                                                               infiniopTensorDescriptor_t x_desc,
                                                                               infiniopTensorDescriptor_t x_packed_desc,
                                                                               infiniopTensorDescriptor_t x_scale_desc);

__INFINI_C __export infiniStatus_t infiniopGetPerTensorDequantFp8WorkspaceSize(infiniopPerTensorDequantFp8Descriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopPerTensorDequantFp8(infiniopPerTensorDequantFp8Descriptor_t desc,
                                                               void *workspace,
                                                               size_t workspace_size,
                                                               void *x,
                                                               const void *x_packed,
                                                               const void *x_scale,
                                                               void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyPerTensorDequantFp8Descriptor(infiniopPerTensorDequantFp8Descriptor_t desc);

#endif
