#include "infinicore/ops/dsv4_sglang_silu_and_mul_clamp.hpp"

#include "../infiniop_impl.hpp"
#include "infiniop/ops/dsv4_sglang_silu_and_mul_clamp.h"

namespace infinicore::op::dsv4_sglang_silu_and_mul_clamp_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Dsv4SglangSiluAndMulClamp, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, output, input;
};

void *plan(Tensor output, const Tensor &input, double limit) {
    size_t seed = hash_combine(output, input, limit);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Dsv4SglangSiluAndMulClamp,
        seed,
        output->desc(), input->desc(), limit);

    INFINIOP_WORKSPACE_TENSOR(workspace, Dsv4SglangSiluAndMulClamp, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(output), graph::GraphTensor(input)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopDsv4SglangSiluAndMulClamp(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->output->data(), planned->input->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Dsv4SglangSiluAndMulClamp, &plan, &run, &cleanup);

} // namespace infinicore::op::dsv4_sglang_silu_and_mul_clamp_impl::infiniop
