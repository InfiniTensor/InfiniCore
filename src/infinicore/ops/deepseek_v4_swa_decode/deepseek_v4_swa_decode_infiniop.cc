#include "infinicore/ops/deepseek_v4_swa_decode.hpp"

#include "../infiniop_impl.hpp"
#include "infiniop/ops/deepseek_v4_swa_decode.h"

namespace infinicore::op::deepseek_v4_swa_decode_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, DeepseekV4SwaDecode, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace;
    graph::GraphTensor y;
    graph::GraphTensor q;
    graph::GraphTensor k;
    graph::GraphTensor attn_sink;
    graph::GraphTensor positions;
};

void *plan(Tensor y,
           const Tensor &q,
           const Tensor &k,
           const Tensor &attn_sink,
           const Tensor &positions,
           size_t key_offset,
           size_t key_len,
           float softmax_scale,
           size_t rope_dim,
           double rope_theta,
           bool use_yarn,
           double yarn_factor,
           double yarn_beta_fast,
           double yarn_beta_slow,
           int64_t yarn_original_seq_len,
           double yarn_extrapolation_factor) {
    size_t seed = hash_combine(y, q, k, attn_sink, positions, key_offset, key_len,
                               softmax_scale, rope_dim, rope_theta, use_yarn, yarn_factor,
                               yarn_beta_fast, yarn_beta_slow,
                               yarn_original_seq_len, yarn_extrapolation_factor);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor,
        descriptor,
        DeepseekV4SwaDecode,
        seed,
        y->desc(),
        q->desc(),
        k->desc(),
        attn_sink->desc(),
        positions->desc(),
        key_offset,
        key_len,
        softmax_scale,
        rope_dim,
        rope_theta,
        use_yarn,
        yarn_factor,
        yarn_beta_fast,
        yarn_beta_slow,
        yarn_original_seq_len,
        yarn_extrapolation_factor);
    INFINIOP_WORKSPACE_TENSOR(workspace, DeepseekV4SwaDecode, descriptor);
    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(y),
        graph::GraphTensor(q),
        graph::GraphTensor(k),
        graph::GraphTensor(attn_sink),
        graph::GraphTensor(positions)};
}

void run(void *planned_meta) {
    auto p = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopDeepseekV4SwaDecode(
        p->descriptor->desc,
        p->workspace->data(),
        p->workspace->numel(),
        p->y->data(),
        p->q->data(),
        p->k->data(),
        p->attn_sink->data(),
        p->positions->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4SwaDecode, &plan, &run, &cleanup);

} // namespace infinicore::op::deepseek_v4_swa_decode_impl::infiniop
