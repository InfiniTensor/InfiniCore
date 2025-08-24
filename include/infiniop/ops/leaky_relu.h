#ifndef __INFINIOP_LEAKY_RELU_API_H__
#define __INFINIOP_LEAKY_RELU_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopLeakyReluDescriptor_t;

/// @brief 创建 LeakyReLU 描述符
/// @param handle      上下文句柄
/// @param desc_ptr    输出的算子描述符
/// @param output      输出张量描述符
/// @param input       输入张量描述符
/// @param negative_slope 负斜率 α，float 类型
__C __export infiniStatus_t infiniopCreateLeakyReluDescriptor(
    infiniopHandle_t handle,
    infiniopLeakyReluDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output,
    infiniopTensorDescriptor_t input);

/// @brief 获取 workspace 大小
__C __export infiniStatus_t infiniopGetLeakyReluWorkspaceSize(
    infiniopLeakyReluDescriptor_t desc,
    size_t *size);

/// @brief 执行 LeakyReLU 运算
__C __export infiniStatus_t infiniopLeakyRelu(
    infiniopLeakyReluDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    float negative_slope,
    void *stream);

/// @brief 销毁 LeakyReLU 描述符
__C __export infiniStatus_t infiniopDestroyLeakyReluDescriptor(
    infiniopLeakyReluDescriptor_t desc);

#endif  // __INFINIOP_LEAKY_RELU_API_H__
