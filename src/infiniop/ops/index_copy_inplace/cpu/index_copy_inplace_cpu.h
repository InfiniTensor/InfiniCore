#ifndef __INFINIOP_INDEX_COPY_INPLACE_CPU_H__
#define __INFINIOP_INDEX_COPY_INPLACE_CPU_H__
//参考rope，它也不是预制件
#include "../index_copy_inplace.h"/*还没创建*/

DESCRIPTOR(cpu)
// 2. 模仿 rope，使用 DESCRIPTOR 宏来声明我们的 Descriptor 类
//    这个宏很可能在 op::index_copy_inplace::cpu 命名空间中，???这个命名空间在哪个文件？
//    为我们定义了 class Descriptor : public op::Descriptor {...};

#endif // __INFINIOP_INDEX_COPY_INPLACE_CPU_H__

//------------------------------空骨架测试用--------------------------------------
// #ifndef __INDEX_COPY_INPLACE_CPU_H__
// #define __INDEX_COPY_INPLACE_CPU_H__

// #include "../operator.cc" // 技巧：为了能拿到 Info 的定义

// // 只在这里【声明】CPU 专属的内核启动函数
// infiniStatus_t index_copy_inplace_kernel_cpu(
//     const IndexCopyInplaceInfo &info,
//     const void *input, void *output, const void *index, void *stream
// );

// #endif