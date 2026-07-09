#ifndef __INFINIOP_DSV4_SGLANG_FUSED_ROPE_H__
#define __INFINIOP_DSV4_SGLANG_FUSED_ROPE_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopDsv4SglangFusedRopeDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateDsv4SglangFusedRopeDescriptor(infiniopHandle_t handle,
                                                                               infiniopDsv4SglangFusedRopeDescriptor_t *desc_ptr,
                                                                               infiniopTensorDescriptor_t q_desc,
                                                                               infiniopTensorDescriptor_t freqs_cis_desc,
                                                                               infiniopTensorDescriptor_t positions_desc,
                                                                               bool inverse);

__INFINI_C __export infiniStatus_t infiniopGetDsv4SglangFusedRopeWorkspaceSize(infiniopDsv4SglangFusedRopeDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopDsv4SglangFusedRope(infiniopDsv4SglangFusedRopeDescriptor_t desc,
                                                               void *workspace,
                                                               size_t workspace_size,
                                                               void *q,
                                                               const void *freqs_cis,
                                                               const void *positions,
                                                               void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyDsv4SglangFusedRopeDescriptor(infiniopDsv4SglangFusedRopeDescriptor_t desc);

#endif
