#include "infinicore/ops/deepseek_v4_compressed_decode.hpp"

#include "../infiniop_impl.hpp"
#include "infiniop/ops/deepseek_v4_compressed_decode.h"

namespace infinicore::op::deepseek_v4_compressed_decode_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, DeepseekV4CompressedDecode, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace;
    graph::GraphTensor y;
    graph::GraphTensor q;
    graph::GraphTensor k;
    graph::GraphTensor kv_comp;
    graph::GraphTensor attn_sink;
    graph::GraphTensor query_positions;
    graph::GraphTensor block_positions;
    graph::GraphTensor indexed_blocks;
};

void *plan(Tensor y,
           const Tensor &q,
           const Tensor &k,
           const Tensor &kv_comp,
           const Tensor &attn_sink,
           const Tensor &query_positions,
           const Tensor &block_positions,
           const Tensor &indexed_blocks,
           size_t key_offset,
           size_t key_len,
           bool causal,
           size_t sliding_window,
           int64_t key_position_base,
           float softmax_scale,
           size_t compress_ratio,
           size_t index_top_k,
           size_t rope_dim,
           double rope_theta,
           bool use_yarn,
           double yarn_factor,
           double yarn_beta_fast,
           double yarn_beta_slow,
           int64_t yarn_original_seq_len,
           double yarn_extrapolation_factor) {
    size_t seed = hash_combine(y, q, k, kv_comp, attn_sink, query_positions,
                               block_positions, indexed_blocks, key_offset, key_len,
                               causal, sliding_window, key_position_base, softmax_scale,
                               compress_ratio, index_top_k,
                               rope_dim, rope_theta, use_yarn, yarn_factor,
                               yarn_beta_fast, yarn_beta_slow,
                               yarn_original_seq_len, yarn_extrapolation_factor);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor,
        descriptor,
        DeepseekV4CompressedDecode,
        seed,
        y->desc(),
        q->desc(),
        k->desc(),
        kv_comp->desc(),
        attn_sink->desc(),
        query_positions->desc(),
        block_positions->desc(),
        indexed_blocks->desc(),
        key_offset,
        key_len,
        causal,
        sliding_window,
        key_position_base,
        softmax_scale,
        compress_ratio,
        index_top_k,
        rope_dim,
        rope_theta,
        use_yarn,
        yarn_factor,
        yarn_beta_fast,
        yarn_beta_slow,
        yarn_original_seq_len,
        yarn_extrapolation_factor);
    INFINIOP_WORKSPACE_TENSOR(workspace, DeepseekV4CompressedDecode, descriptor);
    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(y),
        graph::GraphTensor(q),
        graph::GraphTensor(k),
        graph::GraphTensor(kv_comp),
        graph::GraphTensor(attn_sink),
        graph::GraphTensor(query_positions),
        graph::GraphTensor(block_positions),
        graph::GraphTensor(indexed_blocks)};
}

void run(void *planned_meta) {
    auto p = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopDeepseekV4CompressedDecode(
        p->descriptor->desc,
        p->workspace->data(),
        p->workspace->numel(),
        p->y->data(),
        p->q->data(),
        p->k->data(),
        p->kv_comp->data(),
        p->attn_sink->data(),
        p->query_positions->data(),
        p->block_positions->data(),
        p->indexed_blocks->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4CompressedDecode, &plan, &run, &cleanup);

} // namespace infinicore::op::deepseek_v4_compressed_decode_impl::infiniop
