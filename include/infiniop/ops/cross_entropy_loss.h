#ifndef __INFINIOP_CROSS_ENTROPY_LOSS_API_H__
#define __INFINIOP_CROSS_ENTROPY_LOSS_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopCrossEntropyLossDescriptor_t;

// 1. 创建描述符
__C __export infiniStatus_t infiniopCreateCrossEntropyLossDescriptor(
    infiniopHandle_t handle,
    infiniopCrossEntropyLossDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t loss_desc,
    infiniopTensorDescriptor_t logits_desc,
    infiniopTensorDescriptor_t target_desc);

// 2. 获取工作空间大小
__C __export infiniStatus_t infiniopGetCrossEntropyLossWorkspaceSize(
    infiniopCrossEntropyLossDescriptor_t desc,
    size_t *size);

// 3. 执行计算
__C __export infiniStatus_t infiniopCrossEntropyLoss(
    infiniopCrossEntropyLossDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *loss,
    const void *logits,
    const void *target,
    void *stream);

// 4. 销毁描述符
__C __export infiniStatus_t infiniopDestroyCrossEntropyLossDescriptor(
    infiniopCrossEntropyLossDescriptor_t desc);

#endif // __INFINIOP_CROSS_ENTROPY_LOSS_API_H__