#ifndef __INFINIOP_INDEX_COPY_INPLACE_API_H__
#define __INFINIOP_INDEX_COPY_INPLACE_API_H__

#include "handle.h"
#include "operator_descriptor.h"
#include "tensor_descriptor.h"

typedef struct InfiniopDescriptor *infiniopIndexCopyInplaceDescriptor_t;

__C __export infiniStatus_t infiniopCreateIndexCopyInplaceDescriptor(infiniopHandle_t handle,
                                                        infiniopIndexCopyInplaceDescriptor_t *desc_ptr,//输出参数，用来接收创建好的描述符
                                                        infiniopTensorDescriptor_t input,
                                                        infiniopTensorDescriptor_t output,
                                                        int dim,
                                                        infiniopTensorDescriptor_t index);

//__C __export infiniStatus_t infiniopGetAddWorkspaceSize(infiniopAddDescriptor_t desc, size_t *size);//获取工作空间大小，需要临时内存的时候用，这个算子应该用不到

__C __export infiniStatus_t infiniopIndexCopyInplace(infiniopIndexCopyInplaceDescriptor_t desc,
                                        //void *workspace,
                                        //size_t workspace_size,
                                        const void *input,
                                        void *output,
                                        //const void dim,标量直接值传递，不通过指针；这里不需要了，它在创建描述符时已被记录
                                        const void *index,
                                        void *stream);//需要一个流/队列对象

__C __export infiniStatus_t infiniopDestroyIndexCopyInplaceDescriptor(infiniopIndexCopyInplaceDescriptor_t desc);

#endif

//-----------------------------空骨架测试用---------------------------------------
// #ifndef __INFINIOP_INDEX_COPY_INPLACE_API_H__
// #define __INFINIOP_INDEX_COPY_INPLACE_API_H__

// // 1. 包含所有 C-API 都需要的核心定义
// #include "handle.h"
// #include "operator_descriptor.h"
// #include "tensor_descriptor.h"

// #ifdef __cplusplus
// extern "C" {
// #endif

// // 2. 定义一个不透明的描述符类型
// typedef struct InfiniopDescriptor *infiniopIndexCopyInplaceDescriptor_t;

// // 3. 声明“创建描述符”的 C-API 函数
// __C __export infiniStatus_t infiniopCreateIndexCopyInplaceDescriptor(
//     infiniopHandle_t handle,
//     infiniopIndexCopyInplaceDescriptor_t *desc_ptr, // 输出参数
//     infiniopTensorDescriptor_t input_desc,
//     infiniopTensorDescriptor_t output_desc,
//     int dim,
//     infiniopTensorDescriptor_t index_desc);

// // 4. 声明“执行算子”的 C-API 函数
// __C __export infiniStatus_t infiniopIndexCopyInplace(
//     infiniopIndexCopyInplaceDescriptor_t desc,
//     const void *input,
//     void *output,
//     const void *index,
//     void *stream);

// // 5. 声明“销毁描述符”的 C-API 函数
// __C __export infiniStatus_t infiniopDestroyIndexCopyInplaceDescriptor(
//     infiniopIndexCopyInplaceDescriptor_t desc);

// #ifdef __cplusplus
// } // extern "C"
// #endif

// #endif // __INFINIOP_INDEX_COPY_INPLACE_API_H__