#ifndef __INFINIOP_DEEPSEEK_V4_ROUTER_API_H__
#define __INFINIOP_DEEPSEEK_V4_ROUTER_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopDeepseekV4TopkRouterDescriptor_t;
typedef struct InfiniopDescriptor *infiniopDeepseekV4HashRouterDescriptor_t;
typedef struct InfiniopDescriptor *infiniopDeepseekV4HashTopkRouterDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateDeepseekV4TopkRouterDescriptor(
    infiniopHandle_t handle,
    infiniopDeepseekV4TopkRouterDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t topk_weights_desc,
    infiniopTensorDescriptor_t topk_indices_desc,
    infiniopTensorDescriptor_t logits_desc,
    infiniopTensorDescriptor_t bias_desc,
    bool renormalize);

__INFINI_C __export infiniStatus_t infiniopGetDeepseekV4TopkRouterWorkspaceSize(
    infiniopDeepseekV4TopkRouterDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopDeepseekV4TopkRouter(
    infiniopDeepseekV4TopkRouterDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *topk_weights,
    void *topk_indices,
    const void *logits,
    const void *bias,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyDeepseekV4TopkRouterDescriptor(
    infiniopDeepseekV4TopkRouterDescriptor_t desc);

__INFINI_C __export infiniStatus_t infiniopCreateDeepseekV4HashRouterDescriptor(
    infiniopHandle_t handle,
    infiniopDeepseekV4HashRouterDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t topk_weights_desc,
    infiniopTensorDescriptor_t topk_indices_desc,
    infiniopTensorDescriptor_t logits_desc,
    infiniopTensorDescriptor_t input_ids_desc,
    infiniopTensorDescriptor_t tid2eid_desc,
    bool renormalize);

__INFINI_C __export infiniStatus_t infiniopGetDeepseekV4HashRouterWorkspaceSize(
    infiniopDeepseekV4HashRouterDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopDeepseekV4HashRouter(
    infiniopDeepseekV4HashRouterDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *topk_weights,
    void *topk_indices,
    const void *logits,
    const void *input_ids,
    const void *tid2eid,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyDeepseekV4HashRouterDescriptor(
    infiniopDeepseekV4HashRouterDescriptor_t desc);

__INFINI_C __export infiniStatus_t infiniopCreateDeepseekV4HashTopkRouterDescriptor(
    infiniopHandle_t handle,
    infiniopDeepseekV4HashTopkRouterDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t topk_weights_desc,
    infiniopTensorDescriptor_t topk_indices_desc,
    infiniopTensorDescriptor_t hidden_states_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t input_ids_desc,
    infiniopTensorDescriptor_t tid2eid_desc,
    bool renormalize);

__INFINI_C __export infiniStatus_t infiniopGetDeepseekV4HashTopkRouterWorkspaceSize(
    infiniopDeepseekV4HashTopkRouterDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopDeepseekV4HashTopkRouter(
    infiniopDeepseekV4HashTopkRouterDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *topk_weights,
    void *topk_indices,
    const void *hidden_states,
    const void *weight,
    const void *input_ids,
    const void *tid2eid,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyDeepseekV4HashTopkRouterDescriptor(
    infiniopDeepseekV4HashTopkRouterDescriptor_t desc);

#endif
