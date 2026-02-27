#include "infinicore/ops/swiglu_cuda.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::swiglu_cuda_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, SwiGLUCuda, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace;
    graph::GraphTensor c;
    graph::GraphTensor a;
    graph::GraphTensor b;
};

void *plan(Tensor c, const Tensor &a, const Tensor &b) {
    size_t key = hash_combine(c, a, b);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, SwiGLUCuda,
        key, c->desc(), a->desc(), b->desc());

    INFINIOP_WORKSPACE_TENSOR(workspace, SwiGLUCuda, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(c),
        graph::GraphTensor(a),
        graph::GraphTensor(b)};
}

void run(void *planned_meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(
        infiniopSwiGLUCuda(
            p->descriptor->desc,
            p->workspace->data(),
            p->workspace->numel(),
            p->c->data(),
            p->a->data(),
            p->b->data(),
            context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(SwiGLUCuda, &plan, &run, &cleanup);

} // namespace infinicore::op::swiglu_impl::infiniop
