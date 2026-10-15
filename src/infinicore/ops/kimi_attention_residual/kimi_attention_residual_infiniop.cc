#include "infinicore/ops/kimi_attention_residual.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::kimi_attention_residual_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, KimiAttentionResidual, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace;
    graph::GraphTensor output;
    graph::GraphTensor values;
    graph::GraphTensor norm_weight;
    graph::GraphTensor projection_weight;
};

void *plan(Tensor output,
           const Tensor &values,
           const Tensor &norm_weight,
           const Tensor &projection_weight,
           float epsilon) {
    size_t seed = hash_combine(
        output, values, norm_weight, projection_weight, epsilon);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, KimiAttentionResidual, seed,
        output->desc(), values->desc(), norm_weight->desc(),
        projection_weight->desc(), epsilon);

    INFINIOP_WORKSPACE_TENSOR(workspace, KimiAttentionResidual, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(output),
        graph::GraphTensor(values),
        graph::GraphTensor(norm_weight),
        graph::GraphTensor(projection_weight),
    };
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopKimiAttentionResidual(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->output->data(),
        planned->values->data(),
        planned->norm_weight->data(),
        planned->projection_weight->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(
    KimiAttentionResidual, &plan, &run, &cleanup);

} // namespace infinicore::op::kimi_attention_residual_impl::infiniop
