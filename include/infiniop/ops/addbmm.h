#ifndef INFINIOP_OPS_ADDBMM_H
#define INFINIOP_OPS_ADDBMM_H

#include "../operator_descriptor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct infiniopAddbmmDescriptor *infiniopAddbmmDescriptor_t;

// 创建 Descriptor，包含 alpha 和 beta 参数
__C infiniStatus_t infiniopCreateAddbmmDescriptor(
    infiniopHandle_t handle,
    infiniopAddbmmDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t batch1_desc,
    infiniopTensorDescriptor_t batch2_desc,
    float alpha,
    float beta);

__C infiniStatus_t infiniopGetAddbmmWorkspaceSize(
    infiniopAddbmmDescriptor_t desc, 
    size_t *size);

__C infiniStatus_t infiniopAddbmm(
    infiniopAddbmmDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    const void *batch1,
    const void *batch2,
    void *stream);

__C infiniStatus_t infiniopDestroyAddbmmDescriptor(
    infiniopAddbmmDescriptor_t desc);

#ifdef __cplusplus
}
#endif

#endif // INFINIOP_OPS_ADDBMM_H