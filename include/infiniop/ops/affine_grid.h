#ifndef INFINIOP_OPS_AFFINE_GRID_H
#define INFINIOP_OPS_AFFINE_GRID_H
#include <stdint.h>
#include "../operator_descriptor.h"

#ifdef __cplusplus
extern "C" {
#endif

// 1. 定义 Descriptor 句柄类型
typedef struct infiniopAffineGridDescriptor *infiniopAffineGridDescriptor_t;

// 2. 创建 Descriptor
// 注意：必须包含 align_corners 参数，通常使用 uint8_t 或 bool
__C infiniStatus_t infiniopCreateAffineGridDescriptor(
    infiniopHandle_t handle,
    infiniopAffineGridDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    uint8_t align_corners);

// 3. 获取 Workspace 大小
__C infiniStatus_t infiniopGetAffineGridWorkspaceSize(
    infiniopAffineGridDescriptor_t desc, 
    size_t *size);

// 4. 执行计算
__C infiniStatus_t infiniopAffineGrid(
    infiniopAffineGridDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream);

// 5. 销毁 Descriptor
__C infiniStatus_t infiniopDestroyAffineGridDescriptor(
    infiniopAffineGridDescriptor_t desc);

#ifdef __cplusplus
}
#endif

#endif // INFINIOP_OPS_AFFINE_GRID_H