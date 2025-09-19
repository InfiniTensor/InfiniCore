#include "infinicore/op/matmul.h"

#include <infiniop.h>

namespace infinicore::op {

common::OpDispatcher<Matmul::schema> Matmul::dispatcher;

void Matmul::execute(Tensor c, Tensor a, Tensor b) {
    dispatcher.lookup(context::getDevice().getType())(c, a, b);
}
} // namespace infinicore::op

namespace infinicore::op::matmul_impl {
void infiniop(Tensor c, Tensor a, Tensor b) {
    infiniopGemmDescriptor_t desc;
    // if (!cache_manager->getGemmDescriptor(key, desc)) {
    //     RUN_INFINI(infiniopCreateGemmDescriptor(op_handle, &desc, c->desc(), a->desc(), b->desc()));
    //     cache_manager->putGemmDescriptor(key, desc);
    // }
    infiniopCreateGemmDescriptor(context::getInfiniopHandle(), &desc, c->desc(), a->desc(), b->desc());

    size_t workspace_size = 0;
    infiniopGetGemmWorkspaceSize(desc, &workspace_size);
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    infiniopGemm(
        desc, workspace->data(), workspace_size,
        c->data(), a->data(), b->data(), 1.f, 0.f, context::getStream());
}

static bool registered = []() {
    Matmul::dispatcher.registerAll(infiniop);
    return true;
};
} // namespace infinicore::op::matmul_impl

namespace infinicore::op {
Tensor matmul(Tensor a, Tensor b) {
    auto c = Tensor::empty({a->size(0), b->size(1)}, a->dtype(), a->device());
    matmul_(c, a, b);
    return c;
}
void matmul_(Tensor c, Tensor a, Tensor b) {
    Matmul::execute(c, a, b);
}
} // namespace infinicore::op
