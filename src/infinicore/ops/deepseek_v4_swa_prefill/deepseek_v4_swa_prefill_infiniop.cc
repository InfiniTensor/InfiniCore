#include "infinicore/ops/deepseek_v4_swa_prefill.hpp"

#include "../infiniop_impl.hpp"
#include "infiniop/ops/deepseek_v4_swa_prefill.h"

namespace infinicore::op::deepseek_v4_swa_prefill_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, DeepseekV4SwaPrefill, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace;
    graph::GraphTensor y;
    graph::GraphTensor q;
    graph::GraphTensor k;
    graph::GraphTensor attn_sink;
    graph::GraphTensor query_positions;
    graph::GraphTensor key_positions;
};

void *plan(Tensor y,
           const Tensor &q,
           const Tensor &k,
           const Tensor &attn_sink,
           const Tensor &query_positions,
           const Tensor &key_positions,
           float softmax_scale,
           size_t window,
           size_t rope_dim,
           double rope_theta,
           bool use_yarn,
           double yarn_factor,
           double yarn_beta_fast,
           double yarn_beta_slow,
           int64_t yarn_original_seq_len,
           double yarn_extrapolation_factor) {
    size_t seed = hash_combine(y, q, k, attn_sink, query_positions, key_positions,
                               softmax_scale, window, rope_dim, rope_theta,
                               use_yarn, yarn_factor, yarn_beta_fast, yarn_beta_slow,
                               yarn_original_seq_len, yarn_extrapolation_factor);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor,
        descriptor,
        DeepseekV4SwaPrefill,
        seed,
        y->desc(),
        q->desc(),
        k->desc(),
        attn_sink->desc(),
        query_positions->desc(),
        key_positions->desc(),
        softmax_scale,
        window,
        rope_dim,
        rope_theta,
        use_yarn,
        yarn_factor,
        yarn_beta_fast,
        yarn_beta_slow,
        yarn_original_seq_len,
        yarn_extrapolation_factor);
    INFINIOP_WORKSPACE_TENSOR(workspace, DeepseekV4SwaPrefill, descriptor);
    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(y),
        graph::GraphTensor(q),
        graph::GraphTensor(k),
        graph::GraphTensor(attn_sink),
        graph::GraphTensor(query_positions),
        graph::GraphTensor(key_positions)};
}

void run(void *planned_meta) {
    auto p = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopDeepseekV4SwaPrefill(
        p->descriptor->desc,
        p->workspace->data(),
        p->workspace->numel(),
        p->y->data(),
        p->q->data(),
        p->k->data(),
        p->attn_sink->data(),
        p->query_positions->data(),
        p->key_positions->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4SwaPrefill, &plan, &run, &cleanup);

} // namespace infinicore::op::deepseek_v4_swa_prefill_impl::infiniop
