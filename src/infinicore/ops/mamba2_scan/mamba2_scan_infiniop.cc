#include "../infiniop_impl.hpp"
#include "infinicore/ops/mamba2_scan.hpp"

namespace infinicore::op::mamba2_scan_impl::infiniop {
INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Mamba2Scan, 100);
struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, out, x, dt, b, c, a, d, dt_bias, state, offsets, initial_indices, final_indices;
};
void *plan(Tensor out, const Tensor &x, const Tensor &dt, const Tensor &b, const Tensor &c, const Tensor &a, const Tensor &d, const Tensor &dt_bias, Tensor state, const Tensor &offsets, const Tensor &initial_indices, const Tensor &final_indices) {
    size_t seed = hash_combine(out, x, dt, b, c, a, d, dt_bias, state, offsets, initial_indices, final_indices);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(Descriptor, descriptor, Mamba2Scan, seed, out->desc(), x->desc(), dt->desc(), b->desc(), c->desc(), a->desc(), d->desc(), dt_bias->desc(), state->desc(), offsets->desc(), initial_indices->desc(), final_indices->desc());
    INFINIOP_WORKSPACE_TENSOR(workspace, Mamba2Scan, descriptor);
    return new PlannedMeta{descriptor, graph::GraphTensor(workspace), graph::GraphTensor(out), graph::GraphTensor(x), graph::GraphTensor(dt), graph::GraphTensor(b), graph::GraphTensor(c), graph::GraphTensor(a), graph::GraphTensor(d), graph::GraphTensor(dt_bias), graph::GraphTensor(state), graph::GraphTensor(offsets), graph::GraphTensor(initial_indices), graph::GraphTensor(final_indices)};
}
void run(void *planned_meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopMamba2Scan(p->descriptor->desc, p->workspace->data(), p->workspace->numel(), p->out->data(), p->x->data(), p->dt->data(), p->b->data(), p->c->data(), p->a->data(), p->d->data(), p->dt_bias->data(), p->state->data(), p->offsets->data(), p->initial_indices->data(), p->final_indices->data(), context::getStream()));
}
void cleanup(void **planned_meta) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta);
    *planned_meta = nullptr;
}
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Mamba2Scan, &plan, &run, &cleanup);
} // namespace infinicore::op::mamba2_scan_impl::infiniop
