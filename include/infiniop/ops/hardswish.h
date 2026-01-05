#ifndef __INFINIOP_HARDSWISH_API_H__
#define __INFINIOP_HARDSWISH_API_H__

#include "../operator_descriptor.h"

// 定义 HardSwish 的描述符类型
typedef struct InfiniopDescriptor *infiniopHardSwishDescriptor_t;

// 1. 创建描述符
__C __export infiniStatus_t infiniopCreateHardSwishDescriptor(
    infiniopHandle_t handle,
    infiniopHardSwishDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output,
    infiniopTensorDescriptor_t input);

// 2. 获取 Workspace 大小
__C __export infiniStatus_t infiniopGetHardSwishWorkspaceSize(
    infiniopHardSwishDescriptor_t desc, 
    size_t *size);

// 3. 执行算子
__C __export infiniStatus_t infiniopHardSwish(
    infiniopHardSwishDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream);

// 4. 销毁描述符
__C __export infiniStatus_t infiniopDestroyHardSwishDescriptor(
    infiniopHardSwishDescriptor_t desc);

#endif // __INFINIOP_HARDSWISH_API_H__