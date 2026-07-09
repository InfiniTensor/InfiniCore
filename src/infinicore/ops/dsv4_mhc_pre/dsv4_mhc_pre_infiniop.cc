#include "infinicore/ops/dsv4_mhc_pre.hpp"

#include "../infiniop_impl.hpp"
#include "infiniop/ops/dsv4_mhc_pre.h"

namespace infinicore::op::dsv4_mhc_pre_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Dsv4MhcPre, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, output, input, scale, base;
};

void *plan(Tensor output, const Tensor &input, const Tensor &scale, const Tensor &base, float eps) {
    size_t seed = hash_combine(output, input, scale, base, eps);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Dsv4MhcPre,
        seed,
        output->desc(), input->desc(), scale->desc(), base->desc(), eps);

    INFINIOP_WORKSPACE_TENSOR(workspace, Dsv4MhcPre, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(output),
        graph::GraphTensor(input),
        graph::GraphTensor(scale),
        graph::GraphTensor(base)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopDsv4MhcPre(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->output->data(),
        planned->input->data(),
        planned->scale->data(),
        planned->base->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Dsv4MhcPre, &plan, &run, &cleanup);

} // namespace infinicore::op::dsv4_mhc_pre_impl::infiniop
