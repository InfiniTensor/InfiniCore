#ifndef __INFINIOP_AVG_POOL1D_API_H__
#define __INFINIOP_AVG_POOL1D_API_H__

#include "../operator_descriptor.h"

// 定义 AvgPool1d 的描述符类型
typedef struct InfiniopDescriptor *infiniopAvgPool1dDescriptor_t;

// 1. 创建描述符
__C __export infiniStatus_t infiniopCreateAvgPool1dDescriptor(
    infiniopHandle_t handle,
    infiniopAvgPool1dDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output,
    infiniopTensorDescriptor_t input,
    size_t kernel_size, // 池化核大小
    size_t stride,      // 步长
    size_t padding      // 填充大小
);

// 2. 获取 Workspace 大小
__C __export infiniStatus_t infiniopGetAvgPool1dWorkspaceSize(
    infiniopAvgPool1dDescriptor_t desc, 
    size_t *size);

// 3. 执行算子
__C __export infiniStatus_t infiniopAvgPool1d(
    infiniopAvgPool1dDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream);

// 4. 销毁描述符
__C __export infiniStatus_t infiniopDestroyAvgPool1dDescriptor(
    infiniopAvgPool1dDescriptor_t desc);

#endif // __INFINIOP_AVG_POOL1D_API_H__