#include "infinicore/ops/dsv4_sglang_rmsnorm.hpp"

#include "../infiniop_impl.hpp"
#include "infiniop/ops/dsv4_sglang_rmsnorm.h"

namespace infinicore::op::dsv4_sglang_rmsnorm_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Dsv4SglangRmsnorm, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, output, input;
};

void *plan(Tensor output, const Tensor &input, double eps) {
    size_t seed = hash_combine(output, input, eps);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Dsv4SglangRmsnorm,
        seed,
        output->desc(), input->desc(), eps);

    INFINIOP_WORKSPACE_TENSOR(workspace, Dsv4SglangRmsnorm, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(output),
        graph::GraphTensor(input)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopDsv4SglangRmsnorm(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->output->data(),
        planned->input->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Dsv4SglangRmsnorm, &plan, &run, &cleanup);

} // namespace infinicore::op::dsv4_sglang_rmsnorm_impl::infiniop
