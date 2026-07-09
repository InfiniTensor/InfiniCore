#ifndef __INFINIOP_DSV4_PER_TOKEN_GROUP_QUANT_INT8_H__
#define __INFINIOP_DSV4_PER_TOKEN_GROUP_QUANT_INT8_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopDsv4PerTokenGroupQuantInt8Descriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateDsv4PerTokenGroupQuantInt8Descriptor(infiniopHandle_t handle,
                                                                                      infiniopDsv4PerTokenGroupQuantInt8Descriptor_t *desc_ptr,
                                                                                      infiniopTensorDescriptor_t q_desc,
                                                                                      infiniopTensorDescriptor_t scale_desc,
                                                                                      infiniopTensorDescriptor_t x_desc,
                                                                                      int group_size);

__INFINI_C __export infiniStatus_t infiniopGetDsv4PerTokenGroupQuantInt8WorkspaceSize(infiniopDsv4PerTokenGroupQuantInt8Descriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopDsv4PerTokenGroupQuantInt8(infiniopDsv4PerTokenGroupQuantInt8Descriptor_t desc,
                                                                      void *workspace,
                                                                      size_t workspace_size,
                                                                      void *q,
                                                                      void *scale,
                                                                      const void *x,
                                                                      void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyDsv4PerTokenGroupQuantInt8Descriptor(infiniopDsv4PerTokenGroupQuantInt8Descriptor_t desc);

#endif
