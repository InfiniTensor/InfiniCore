// #include "gu_topk_softmax.h"
// #include "infinicore/context/context.hpp"
// // 必须包含底层的 C 接口定义
// #include "infiniop/ops/topksoftmax.h" 

// namespace infinicore::op {

// std::pair<Tensor, Tensor> topk_softmax(Tensor input, int k, bool normalize, infiniopHandle_t handle) {
//     // 1. 手动计算输出形状
//     // input: [..., Hidden] -> output: [..., K]
//     // 假设 TopK 操作在最后一维进行
//     Shape out_shape = input->shape();
//     out_shape[out_shape.size() - 1] = k; 

//     // 2. 创建输出 Tensors (分配物理内存)
//     // values (概率值) 类型与 input 一致
//     Tensor values = Tensor::empty(out_shape, input->dtype(), input->device());
//     // indices (索引) 类型通常是 int32 (从源码 (int *)indices 推断)
//     Tensor indices = Tensor::empty(out_shape, DataType::I32, input->device());

//     // 3. 创建描述符 (Create Descriptor)
//     // 根据源码: infiniopCreateTopksoftmaxDescriptor(handle, &desc_ptr, x_desc)
//     // 只传入了 input 的描述符
//     infiniopTopksoftmaxDescriptor_t desc;
//     infiniopCreateTopksoftmaxDescriptor(
//         handle,
//         &desc,
//         input->desc() 
//     );

//     // 4. 申请 Workspace
//     size_t workspace_size = 0;
//     infiniopGetTopksoftmaxWorkspaceSize(desc, &workspace_size);
    
//     // 使用智能指针管理内存 (RAII)
//     std::shared_ptr<infinicore::Memory> workspace_mem = nullptr;
//     void* workspace = nullptr;

//     if (workspace_size > 0) {
//         workspace_mem = context::allocateMemory(workspace_size);
//         workspace = workspace_mem->data();
//     }

//     // 5. 执行计算 (Calculate)
//     // 根据源码: infiniopTopksoftmax(desc, ws, ws_size, val, idx, x, topk, norm, stream)
//     // topk 和 norm 是在这里传入的！
//     void* stream = nullptr; 
//     // stream = context::getStream();

//     infiniopTopksoftmax(
//         desc,
//         workspace,
//         workspace_size,
//         values->data(),  // void* values
//         indices->data(), // void* indices
//         input->data(),   // const void* x
//         static_cast<size_t>(k),     // const size_t topk
//         normalize ? 1 : 0,          // const int norm (0 or 1)
//         stream
//     );

//     // 6. 销毁描述符
//     infiniopDestroyTopksoftmaxDescriptor(desc);

//     return {values, indices};
// }

// } // namespace infinicore::op

#include "gu_topk_softmax.h"
#include "infinicore/context/context.hpp"
#include "infiniop/ops/topksoftmax.h" 
#include "infinirt.h" // 用于同步
#include <iostream>

namespace infinicore::op {

// 辅助宏：检查状态，出错则抛异常
#define CHECK_STATUS(call, msg) \
    do { \
        auto status = (call); \
        if (status != INFINI_STATUS_SUCCESS) { \
            std::string err_msg = std::string("[TopK_Softmax Error] ") + msg + " (Status Code: " + std::to_string(status) + ")"; \
            std::cerr << err_msg << std::endl; \
            throw std::runtime_error(err_msg); \
        } \
    } while (0)

std::pair<Tensor, Tensor> topk_softmax(Tensor input, int k, bool normalize, infiniopHandle_t handle) {
    Shape out_shape = input->shape();
    out_shape[out_shape.size() - 1] = k; 

    // 1. 创建输出张量
    // values: 概率
    Tensor values = Tensor::empty(out_shape, input->dtype(), input->device());
    // indices: 索引 (I32)
    Tensor indices = Tensor::empty(out_shape, DataType::I32, input->device());

    // 2. 创建算子描述符
    infiniopTopksoftmaxDescriptor_t desc;
    CHECK_STATUS(
        infiniopCreateTopksoftmaxDescriptor(handle, &desc, input->desc()),
        "Failed to create descriptor"
    );

    // 3. 申请 Workspace
    size_t workspace_size = 0;
    CHECK_STATUS(
        infiniopGetTopksoftmaxWorkspaceSize(desc, &workspace_size),
        "Failed to get workspace size"
    );
    
    std::shared_ptr<infinicore::Memory> workspace_mem = nullptr;
    void* workspace = nullptr;
    if (workspace_size > 0) {
        workspace_mem = context::allocateMemory(workspace_size);
        workspace = workspace_mem->data();
    }

    void* stream = nullptr; 

    // 4. 执行计算 (Execute)
    // 注意：如果这里失败，会直接抛出异常，而不是返回全0
    CHECK_STATUS(
        infiniopTopksoftmax(
            desc,
            workspace,
            workspace_size,
            values->data(),  // Arg 1: Values (Probs) - 按照你的意愿保持原样
            indices->data(), // Arg 2: Indices (Ints) - 按照你的意愿保持原样
            input->data(),   
            static_cast<size_t>(k),
            normalize ? 1 : 0,
            stream
        ),
        "Kernel execution failed"
    );

    // 5. 销毁描述符 (防止资源泄漏)
    CHECK_STATUS(
        infiniopDestroyTopksoftmaxDescriptor(desc),
        "Failed to destroy descriptor"
    );

    // 6. 【关键】强制同步
    // 防止因为 GPU 还没算完，后续代码就去读，导致读到 0
    infinirtDeviceSynchronize();

    return {values, indices};
}

} // namespace infinicore::op