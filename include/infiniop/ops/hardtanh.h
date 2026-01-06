#ifndef __INFINIOP_HARDTANH_API_H__
#define __INFINIOP_HARDTANH_API_H__

#include "../operator_descriptor.h"

// 定义 HardTanh 的 Descriptor 类型
typedef struct InfiniopDescriptor *infiniopHardTanhDescriptor_t;

/**
 * @brief 创建 HardTanh 算子描述符
 * @param min_val 截断的最小值 (通常为 -1.0)
 * @param max_val 截断的最大值 (通常为 1.0)
 */
__C __export infiniStatus_t infiniopCreateHardTanhDescriptor(infiniopHandle_t handle,
                                                          infiniopHardTanhDescriptor_t *desc_ptr,
                                                          infiniopTensorDescriptor_t output,
                                                          infiniopTensorDescriptor_t input,
                                                          float min_val,
                                                          float max_val);

/**
 * @brief 获取算子所需的临时工作空间大小
 */
__C __export infiniStatus_t infiniopGetHardTanhWorkspaceSize(infiniopHardTanhDescriptor_t desc, 
                                                           size_t *size);

/**
 * @brief 执行 HardTanh 算子
 */
__C __export infiniStatus_t infiniopHardTanh(infiniopHardTanhDescriptor_t desc,
                                           void *workspace,
                                           size_t workspace_size,
                                           void *output,
                                           const void *input,
                                           void *stream);

/**
 * @brief 销毁描述符并释放相关资源
 */
__C __export infiniStatus_t infiniopDestroyHardTanhDescriptor(infiniopHardTanhDescriptor_t desc);

#endif