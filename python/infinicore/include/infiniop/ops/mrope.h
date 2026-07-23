#ifndef __INFINIOP_MROPE_API_H__
#define __INFINIOP_MROPE_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopMRoPEDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateMRoPEDescriptor(
    infiniopHandle_t handle,
    infiniopMRoPEDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t q_out,
    infiniopTensorDescriptor_t k_out,
    infiniopTensorDescriptor_t q,
    infiniopTensorDescriptor_t k,
    infiniopTensorDescriptor_t cos,
    infiniopTensorDescriptor_t sin,
    infiniopTensorDescriptor_t positions,
    int head_size,
    int rotary_dim,
    int section_t,
    int section_h,
    int section_w,
    bool interleaved);

__INFINI_C __export infiniStatus_t infiniopGetMRoPEWorkspaceSize(infiniopMRoPEDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopMRoPE(
    infiniopMRoPEDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *q_out,
    void *k_out,
    const void *q,
    const void *k,
    void const *cos,
    void const *sin,
    void const *positions,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyMRoPEDescriptor(infiniopMRoPEDescriptor_t desc);

#endif // __INFINIOP_MROPE_API_H__
