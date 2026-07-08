#ifndef __INFINIOP_DEEPSEEK_V4_MHC_POST_API_H__
#define __INFINIOP_DEEPSEEK_V4_MHC_POST_API_H__

#include "../operator_descriptor.h"

#if defined(__cplusplus)
extern "C" {
#endif

typedef struct InfiniopDescriptor *infiniopDeepseekV4MHCPostDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateDeepseekV4MHCPostDescriptor(
    infiniopHandle_t handle,
    infiniopDeepseekV4MHCPostDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t new_x_desc,
    infiniopTensorDescriptor_t residual_desc,
    infiniopTensorDescriptor_t post_desc,
    infiniopTensorDescriptor_t comb_desc);

__INFINI_C __export infiniStatus_t infiniopGetDeepseekV4MHCPostWorkspaceSize(
    infiniopDeepseekV4MHCPostDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopDeepseekV4MHCPost(
    infiniopDeepseekV4MHCPostDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *new_x,
    const void *residual,
    const void *post,
    const void *comb,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyDeepseekV4MHCPostDescriptor(
    infiniopDeepseekV4MHCPostDescriptor_t desc);

#if defined(__cplusplus)
}
#endif

#endif
