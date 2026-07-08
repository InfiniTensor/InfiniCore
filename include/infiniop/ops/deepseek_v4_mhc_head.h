#ifndef __INFINIOP_DEEPSEEK_V4_MHC_HEAD_API_H__
#define __INFINIOP_DEEPSEEK_V4_MHC_HEAD_API_H__

#include "../operator_descriptor.h"

#if defined(__cplusplus)
extern "C" {
#endif

typedef struct InfiniopDescriptor *infiniopDeepseekV4MHCHeadCollapseDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateDeepseekV4MHCHeadCollapseDescriptor(
    infiniopHandle_t handle,
    infiniopDeepseekV4MHCHeadCollapseDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t mixes_desc,
    infiniopTensorDescriptor_t base_desc,
    infiniopTensorDescriptor_t scale_desc,
    float epsilon);

__INFINI_C __export infiniStatus_t infiniopGetDeepseekV4MHCHeadCollapseWorkspaceSize(
    infiniopDeepseekV4MHCHeadCollapseDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopDeepseekV4MHCHeadCollapse(
    infiniopDeepseekV4MHCHeadCollapseDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    const void *mixes,
    const void *base,
    const void *scale,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyDeepseekV4MHCHeadCollapseDescriptor(
    infiniopDeepseekV4MHCHeadCollapseDescriptor_t desc);

#if defined(__cplusplus)
}
#endif

#endif
