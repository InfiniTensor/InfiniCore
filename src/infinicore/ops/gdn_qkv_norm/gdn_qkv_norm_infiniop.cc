#include "../infiniop_impl.hpp"
#include "infinicore/ops/gdn_qkv_norm.hpp"
#include "infiniop/ops/gdn_qkv_norm.h"
namespace infinicore::op::gdn_qkv_norm_impl::infiniop {
INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, GDNQKVNorm, 100);
struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace;
    graph::GraphTensor q_out;
    graph::GraphTensor k_out;
    graph::GraphTensor v_out;
    graph::GraphTensor q;
    graph::GraphTensor k;
    graph::GraphTensor v;
};
void *plan(Tensor q_out, Tensor k_out, Tensor v_out, const Tensor &q, const Tensor &k, const Tensor &v) {
    size_t key = hash_combine(q_out, k_out, v_out, q, k, v);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(Descriptor, descriptor, GDNQKVNorm, key, q_out->desc(), k_out->desc(), v_out->desc(), q->desc(), k->desc(), v->desc());
    INFINIOP_WORKSPACE_TENSOR(workspace, GDNQKVNorm, descriptor);
    return new PlannedMeta{descriptor, graph::GraphTensor(workspace), graph::GraphTensor(q_out), graph::GraphTensor(k_out), graph::GraphTensor(v_out), graph::GraphTensor(q), graph::GraphTensor(k), graph::GraphTensor(v)};
}
void run(void *meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(meta);
    INFINICORE_CHECK_ERROR(infiniopGDNQKVNorm(p->descriptor->desc, p->workspace->data(), p->workspace->numel(), p->q_out->data(), p->k_out->data(), p->v_out->data(), p->q->data(), p->k->data(), p->v->data(), context::getStream()));
}
void cleanup(void **meta) {
    delete *reinterpret_cast<PlannedMeta **>(meta);
    *meta = nullptr;
}
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(GDNQKVNorm, &plan, &run, &cleanup);
} // namespace infinicore::op::gdn_qkv_norm_impl::infiniop
