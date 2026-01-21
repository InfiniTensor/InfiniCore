#ifndef __INFINIOP_SILU_AND_MUL_API_H__
#define __INFINIOP_SILU_AND_MUL_API_H__

#include "../operator_descriptor.h"

// 定义描述符类型
typedef struct InfiniopDescriptor *infiniopSiluAndMulDescriptor_t;

/**
 * @brief 创建 SiluAndMul 算子描述符
 * * 公式: output = silu(input_front) * input_back
 * 其中 input 形状为 [..., 2*d], output 形状为 [..., d]
 */
__C __export infiniStatus_t infiniopCreateSiluAndMulDescriptor(
    infiniopHandle_t handle,
    infiniopSiluAndMulDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output,
    infiniopTensorDescriptor_t input);

/**
 * @brief 获取算子执行所需的临时空间大小
 */
__C __export infiniStatus_t infiniopGetSiluAndMulWorkspaceSize(
    infiniopSiluAndMulDescriptor_t desc, 
    size_t *size);

/**
 * @brief 执行 SiluAndMul 计算
 * * @param workspace 临时空间指针
 * @param workspace_size 临时空间大小
 * @param output 输出张量数据指针 [..., d]
 * @param input 输入张量数据指针 [..., 2*d]
 * @param stream 硬件流指针 (如 musaStream_t)
 */
__C __export infiniStatus_t infiniopSiluAndMul(
    infiniopSiluAndMulDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream);

/**
 * @brief 销毁描述符并释放相关资源
 */
__C __export infiniStatus_t infiniopDestroySiluAndMulDescriptor(
    infiniopSiluAndMulDescriptor_t desc);

#endif // __INFINIOP_SILU_AND_MUL_API_H__
