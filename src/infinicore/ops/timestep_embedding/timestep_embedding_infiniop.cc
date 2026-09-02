#include "../infiniop_impl.hpp"
#include "infinicore/ops/timestep_embedding.hpp"

namespace infinicore::op::timestep_embedding_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, TimestepEmbedding, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor output, timestep;
    float max_period;
};

void *plan(Tensor output, const Tensor &timestep, float max_period) {
    size_t seed = hash_combine(output, timestep);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, TimestepEmbedding,
        seed, output->desc(), timestep->desc());

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(output),
        graph::GraphTensor(timestep),
        max_period};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopTimestepEmbedding(
        planned->descriptor->desc,
        planned->output->data(),
        planned->timestep->data(),
        planned->max_period,
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(
    TimestepEmbedding, &plan, &run, &cleanup);

} // namespace infinicore::op::timestep_embedding_impl::infiniop
