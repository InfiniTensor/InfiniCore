#ifndef __INFINIOP_EQUAL_API_H__
#define __INFINIOP_EQUAL_API_H__

#include "../operator_descriptor.h"

// 定义 Equal 算子的描述符句柄
typedef struct InfiniopDescriptor *infiniopEqualDescriptor_t;

// 1. 创建描述符
// c = a == b
__C __export infiniStatus_t infiniopCreateEqualDescriptor(
    infiniopHandle_t handle,
    infiniopEqualDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t c,  // Output (Result)
    infiniopTensorDescriptor_t a,  // Input A
    infiniopTensorDescriptor_t b); // Input B

// 2. 获取 Workspace 大小
__C __export infiniStatus_t infiniopGetEqualWorkspaceSize(
    infiniopEqualDescriptor_t desc, 
    size_t *size);

// 3. 执行算子
__C __export infiniStatus_t infiniopEqual(
    infiniopEqualDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *c,            // Output data pointer
    const void *a,      // Input A data pointer
    const void *b,      // Input B data pointer
    void *stream);

// 4. 销毁描述符
__C __export infiniStatus_t infiniopDestroyEqualDescriptor(
    infiniopEqualDescriptor_t desc);

#endif // __INFINIOP_EQUAL_API_H__