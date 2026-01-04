#ifndef __INFINIOP_CROSS_ENTROPY_API_H__
#define __INFINIOP_CROSS_ENTROPY_API_H__

#include "../operator_descriptor.h"

// 定义 Cross Entropy 的描述符类型
typedef struct InfiniopDescriptor *infiniopCrossEntropyDescriptor_t;

// 1. 创建描述符
// 相比 Softmax，这里多了一个 target_desc，用于描述标签（Label）的 Shape 和 Dtype
__C __export infiniStatus_t infiniopCreateCrossEntropyDescriptor(
    infiniopHandle_t handle,
    infiniopCrossEntropyDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,      // 输出: Loss (通常是 [Batch, SeqLen] 或 Scalar)
    infiniopTensorDescriptor_t x_desc,      // 输入: Logits (通常是 [Batch, SeqLen, VocabSize])
    infiniopTensorDescriptor_t target_desc  // 输入: Labels (通常是 [Batch, SeqLen]，类型为 int64/int32)
);

// 2. 获取 Workspace 大小
// CE 计算通常也需要临时空间来存储 LogSumExp 的中间结果
__C __export infiniStatus_t infiniopGetCrossEntropyWorkspaceSize(
    infiniopCrossEntropyDescriptor_t desc, 
    size_t *size
);

// 3. 执行算子
// 参数中增加了 target 的数据指针
__C __export infiniStatus_t infiniopCrossEntropy(
    infiniopCrossEntropyDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,            // 输出 Loss 的数据指针
    const void *x,      // 输入 Logits 的数据指针
    const void *target, // 输入 Labels 的数据指针
    void *stream
);

// 4. 销毁描述符
__C __export infiniStatus_t infiniopDestroyCrossEntropyDescriptor(
    infiniopCrossEntropyDescriptor_t desc
);

#endif // __INFINIOP_CROSS_ENTROPY_API_H__