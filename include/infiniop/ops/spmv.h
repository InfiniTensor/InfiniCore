#ifndef __INFINIOP_SPMV_API_H__
#define __INFINIOP_SPMV_API_H__

#include "../operator_descriptor.h"
#include <cstddef>

typedef struct InfiniopDescriptor *infiniopSpMVDescriptor_t;

__C __export infiniStatus_t infiniopCreateSpMVDescriptor(
    infiniopHandle_t handle,
    infiniopSpMVDescriptor_t *desc_ptr,
    size_t num_cols,      // 矩阵列数
    size_t num_rows,      // 行偏移数组长度
    size_t nnz,           // 非零元素数量
    infiniDtype_t dtype); // 数据类型（当前仅支持F32）

__C __export infiniStatus_t infiniopSpMV(
    infiniopSpMVDescriptor_t desc,
    void *y,                 // 输出向量
    const void *x,           // 输入向量
    const void *values,      // 非零元素值数组
    const void *row_ptr,     // 行偏移数组
    const void *col_indices, // 列索引数组
    void *stream);           // 计算流

__C __export infiniStatus_t infiniopDestroySpMVDescriptor(infiniopSpMVDescriptor_t desc);

#endif
