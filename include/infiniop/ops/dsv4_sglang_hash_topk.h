#ifndef __INFINIOP_DSV4_SGLANG_HASH_TOPK_H__
#define __INFINIOP_DSV4_SGLANG_HASH_TOPK_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopDsv4SglangHashTopkDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateDsv4SglangHashTopkDescriptor(infiniopHandle_t handle,
                                                                              infiniopDsv4SglangHashTopkDescriptor_t *desc_ptr,
                                                                              infiniopTensorDescriptor_t router_logits_desc,
                                                                              infiniopTensorDescriptor_t input_ids_desc,
                                                                              infiniopTensorDescriptor_t tid2eid_desc,
                                                                              infiniopTensorDescriptor_t topk_weights_desc,
                                                                              infiniopTensorDescriptor_t topk_ids_desc,
                                                                              float routed_scaling_factor);

__INFINI_C __export infiniStatus_t infiniopGetDsv4SglangHashTopkWorkspaceSize(infiniopDsv4SglangHashTopkDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopDsv4SglangHashTopk(infiniopDsv4SglangHashTopkDescriptor_t desc,
                                                              void *workspace,
                                                              size_t workspace_size,
                                                              const void *router_logits,
                                                              const void *input_ids,
                                                              const void *tid2eid,
                                                              void *topk_weights,
                                                              void *topk_ids,
                                                              void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyDsv4SglangHashTopkDescriptor(infiniopDsv4SglangHashTopkDescriptor_t desc);

#endif
