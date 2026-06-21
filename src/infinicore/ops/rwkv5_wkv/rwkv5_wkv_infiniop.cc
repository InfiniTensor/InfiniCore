#include "infinicore/ops/rwkv5_wkv.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::rwkv5_wkv_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Rwkv5Wkv, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, out, receptance, key, value, time_decay, time_faaaa, state;
};

void *plan(Tensor out,
           const Tensor &receptance,
           const Tensor &key,
           const Tensor &value,
           const Tensor &time_decay,
           const Tensor &time_faaaa,
           Tensor state) {
    size_t seed = hash_combine(out, receptance, key, value, time_decay, time_faaaa, state);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Rwkv5Wkv,
        seed,
        out->desc(),
        receptance->desc(),
        key->desc(),
        value->desc(),
        time_decay->desc(),
        time_faaaa->desc(),
        state->desc());

    INFINIOP_WORKSPACE_TENSOR(workspace, Rwkv5Wkv, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(out),
        graph::GraphTensor(receptance),
        graph::GraphTensor(key),
        graph::GraphTensor(value),
        graph::GraphTensor(time_decay),
        graph::GraphTensor(time_faaaa),
        graph::GraphTensor(state)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopRwkv5Wkv(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->out->data(),
        planned->receptance->data(),
        planned->key->data(),
        planned->value->data(),
        planned->time_decay->data(),
        planned->time_faaaa->data(),
        planned->state->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Rwkv5Wkv, &plan, &run, &cleanup);

} // namespace infinicore::op::rwkv5_wkv_impl::infiniop
