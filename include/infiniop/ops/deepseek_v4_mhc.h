#ifndef __INFINIOP_DEEPSEEK_V4_MHC_API_H__
#define __INFINIOP_DEEPSEEK_V4_MHC_API_H__

#include "../operator_descriptor.h"

#include <stddef.h>

#if defined(__cplusplus)
extern "C" {
#endif

typedef struct InfiniopDescriptor *infiniopDeepseekV4MHCParamsDescriptor_t;
typedef struct InfiniopDescriptor *infiniopDeepseekV4MHCPreCollapseDescriptor_t;
typedef struct InfiniopDescriptor *infiniopDeepseekV4MHCScaleMixesDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateDeepseekV4MHCParamsDescriptor(
    infiniopHandle_t handle,
    infiniopDeepseekV4MHCParamsDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t pre_desc,
    infiniopTensorDescriptor_t post_desc,
    infiniopTensorDescriptor_t comb_desc,
    infiniopTensorDescriptor_t mixes_desc,
    infiniopTensorDescriptor_t base_desc,
    infiniopTensorDescriptor_t scale_desc,
    size_t sinkhorn_iters,
    float epsilon);

__INFINI_C __export infiniStatus_t infiniopGetDeepseekV4MHCParamsWorkspaceSize(
    infiniopDeepseekV4MHCParamsDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopDeepseekV4MHCParams(
    infiniopDeepseekV4MHCParamsDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *pre,
    void *post,
    void *comb,
    const void *mixes,
    const void *base,
    const void *scale,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyDeepseekV4MHCParamsDescriptor(
    infiniopDeepseekV4MHCParamsDescriptor_t desc);

__INFINI_C __export infiniStatus_t infiniopCreateDeepseekV4MHCPreCollapseDescriptor(
    infiniopHandle_t handle,
    infiniopDeepseekV4MHCPreCollapseDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t collapsed_desc,
    infiniopTensorDescriptor_t post_desc,
    infiniopTensorDescriptor_t comb_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t mixes_desc,
    infiniopTensorDescriptor_t base_desc,
    infiniopTensorDescriptor_t scale_desc,
    size_t sinkhorn_iters,
    float epsilon);

__INFINI_C __export infiniStatus_t infiniopGetDeepseekV4MHCPreCollapseWorkspaceSize(
    infiniopDeepseekV4MHCPreCollapseDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopDeepseekV4MHCPreCollapse(
    infiniopDeepseekV4MHCPreCollapseDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *collapsed,
    void *post,
    void *comb,
    const void *x,
    const void *mixes,
    const void *base,
    const void *scale,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyDeepseekV4MHCPreCollapseDescriptor(
    infiniopDeepseekV4MHCPreCollapseDescriptor_t desc);

__INFINI_C __export infiniStatus_t infiniopCreateDeepseekV4MHCScaleMixesDescriptor(
    infiniopHandle_t handle,
    infiniopDeepseekV4MHCScaleMixesDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t scaled_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t raw_mixes_desc,
    float epsilon);

__INFINI_C __export infiniStatus_t infiniopGetDeepseekV4MHCScaleMixesWorkspaceSize(
    infiniopDeepseekV4MHCScaleMixesDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopDeepseekV4MHCScaleMixes(
    infiniopDeepseekV4MHCScaleMixesDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *scaled,
    const void *x,
    const void *raw_mixes,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyDeepseekV4MHCScaleMixesDescriptor(
    infiniopDeepseekV4MHCScaleMixesDescriptor_t desc);

#if defined(__cplusplus)
}
#endif

#endif
