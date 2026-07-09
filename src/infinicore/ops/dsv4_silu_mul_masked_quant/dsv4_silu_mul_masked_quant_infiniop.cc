#include "infinicore/ops/dsv4_silu_mul_masked_quant.hpp"

#include "../infiniop_impl.hpp"
#include "infiniop/ops/dsv4_silu_mul_masked_quant.h"

namespace infinicore::op::dsv4_silu_mul_masked_quant_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Dsv4SiluMulMaskedQuant, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, q, scale, gate, up, mask;
};

void *plan(Tensor q, Tensor scale, const Tensor &gate, const Tensor &up, const Tensor &mask) {
    size_t seed = hash_combine(q, scale, gate, up, mask);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Dsv4SiluMulMaskedQuant,
        seed,
        q->desc(), scale->desc(), gate->desc(), up->desc(), mask->desc());

    INFINIOP_WORKSPACE_TENSOR(workspace, Dsv4SiluMulMaskedQuant, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(q),
        graph::GraphTensor(scale),
        graph::GraphTensor(gate),
        graph::GraphTensor(up),
        graph::GraphTensor(mask)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopDsv4SiluMulMaskedQuant(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->q->data(),
        planned->scale->data(),
        planned->gate->data(),
        planned->up->data(),
        planned->mask->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Dsv4SiluMulMaskedQuant, &plan, &run, &cleanup);

} // namespace infinicore::op::dsv4_silu_mul_masked_quant_impl::infiniop
