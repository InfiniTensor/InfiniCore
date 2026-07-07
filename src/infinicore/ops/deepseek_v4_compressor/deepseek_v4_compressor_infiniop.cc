#include "infinicore/ops/deepseek_v4_compressor.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::deepseek_v4_compressor_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, DeepseekV4Compressor, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace;
    graph::GraphTensor out;
    graph::GraphTensor kv;
    graph::GraphTensor score;
    graph::GraphTensor ape;
    graph::GraphTensor norm_weight;
};

void *plan(Tensor out,
           const Tensor &kv,
           const Tensor &score,
           const Tensor &ape,
           const Tensor &norm_weight,
           size_t compress_ratio,
           float epsilon) {
    size_t seed = hash_combine(out, kv, score, ape, norm_weight, compress_ratio, epsilon);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor,
        descriptor,
        DeepseekV4Compressor,
        seed,
        out->desc(),
        kv->desc(),
        score->desc(),
        ape->desc(),
        norm_weight->desc(),
        compress_ratio,
        epsilon);
    INFINIOP_WORKSPACE_TENSOR(workspace, DeepseekV4Compressor, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(out),
        graph::GraphTensor(kv),
        graph::GraphTensor(score),
        graph::GraphTensor(ape),
        graph::GraphTensor(norm_weight)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopDeepseekV4Compressor(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->out->data(),
        planned->kv->data(),
        planned->score->data(),
        planned->ape->data(),
        planned->norm_weight->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4Compressor, &plan, &run, cleanup);

} // namespace infinicore::op::deepseek_v4_compressor_impl::infiniop
