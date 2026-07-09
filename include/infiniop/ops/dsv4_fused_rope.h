#ifndef __INFINIOP_DSV4_FUSED_ROPE_API_H__
#define __INFINIOP_DSV4_FUSED_ROPE_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopDsv4FusedRopeDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateDsv4FusedRopeDescriptor(infiniopHandle_t handle, infiniopDsv4FusedRopeDescriptor_t *desc_ptr, infiniopTensorDescriptor_t q_desc, infiniopTensorDescriptor_t k_desc, infiniopTensorDescriptor_t freq_real_desc, infiniopTensorDescriptor_t freq_imag_desc, int has_k);
__INFINI_C __export infiniStatus_t infiniopGetDsv4FusedRopeWorkspaceSize(infiniopDsv4FusedRopeDescriptor_t desc, size_t *size);
__INFINI_C __export infiniStatus_t infiniopDsv4FusedRope(infiniopDsv4FusedRopeDescriptor_t desc, void *workspace, size_t workspace_size, void *q, void *k, const void *freq_real, const void *freq_imag, void *stream);
__INFINI_C __export infiniStatus_t infiniopDestroyDsv4FusedRopeDescriptor(infiniopDsv4FusedRopeDescriptor_t desc);

#endif // __INFINIOP_DSV4_FUSED_ROPE_API_H__
