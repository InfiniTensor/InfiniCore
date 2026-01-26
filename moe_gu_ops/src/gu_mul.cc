#include "infinicore/tensor.hpp"
#include "infinicore/context/context.hpp"
#include "infiniop/ops/mul.h" 
#include "gu_mul.h"

namespace infinicore::op {

// 【修改点 1】函数签名增加 infiniopHandle_t handle
Tensor mul(Tensor a, Tensor b, infiniopHandle_t handle) {

    Tensor c = Tensor::empty(a->shape(), a->dtype(), a->device());

    infiniopMulDescriptor_t desc;
    infiniopCreateMulDescriptor(
        handle,
        &desc,
        c->desc(), 
        a->desc(), 
        b->desc()
    );

    size_t workspace_size = 0;
    std::shared_ptr<infinicore::Memory> workspace_mem = nullptr;
    void* workspace = nullptr;
    infiniopGetMulWorkspaceSize(desc, &workspace_size);
    
    if (workspace_size > 0) {
        workspace_mem = context::allocateMemory(workspace_size);
        workspace = workspace_mem->data();
    }

    void* stream = nullptr; 
    
    infiniopMul(
        desc,
        workspace,
        workspace_size,
        c->data(),
        a->data(),
        b->data(),
        stream
    );

    infiniopDestroyMulDescriptor(desc);
    
    return c;
}

} // namespace infinicore::op