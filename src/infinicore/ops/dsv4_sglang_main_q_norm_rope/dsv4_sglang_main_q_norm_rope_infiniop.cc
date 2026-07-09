#include "infinicore/ops/dsv4_sglang_main_q_norm_rope.hpp"

#include "../infiniop_impl.hpp"
#include "infiniop/ops/dsv4_sglang_main_q_norm_rope.h"

namespace infinicore::op::dsv4_sglang_main_q_norm_rope_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Dsv4SglangMainQNormRope, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, output, input, freqs, positions;
};

void *plan(Tensor output, const Tensor &input, const Tensor &freqs, const Tensor &positions, double eps) {
    size_t seed = hash_combine(output, input, freqs, positions, eps);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Dsv4SglangMainQNormRope,
        seed,
        output->desc(), input->desc(), freqs->desc(), positions->desc(), eps);

    INFINIOP_WORKSPACE_TENSOR(workspace, Dsv4SglangMainQNormRope, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(output), graph::GraphTensor(input), graph::GraphTensor(freqs), graph::GraphTensor(positions)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopDsv4SglangMainQNormRope(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->output->data(), planned->input->data(), planned->freqs->data(), planned->positions->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Dsv4SglangMainQNormRope, &plan, &run, &cleanup);

} // namespace infinicore::op::dsv4_sglang_main_q_norm_rope_impl::infiniop
