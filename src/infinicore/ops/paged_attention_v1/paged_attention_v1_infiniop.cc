#include "infinicore/ops/paged_attention_v1.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::paged_attention_v1_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, PagedAttentionV1, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace;
    graph::GraphTensor out;
    graph::GraphTensor query;
    graph::GraphTensor key_cache;
    graph::GraphTensor value_cache;
    graph::GraphTensor block_tables;
    graph::GraphTensor seq_lens;
    std::optional<graph::GraphTensor> alibi_slopes;
    graph::GraphTensor k_scale;
    graph::GraphTensor v_scale;
    int64_t num_kv_heads;
    double scale;
    int64_t block_size;
    int64_t max_seq_len;
    std::string kv_cache_dtype;
    int64_t tp_rank;
    int64_t blocksparse_local_blocks;
    int64_t blocksparse_vert_stride;
    int64_t blocksparse_block_size;
    int64_t blocksparse_head_sliding_step;
};

void *plan(Tensor &out,          // [num_seqs, num_heads, head_size]
           Tensor &query,        // [num_seqs, num_heads, head_size]
           Tensor &key_cache,    // [num_blocks, num_heads, head_size/x, block_size, x]
           Tensor &value_cache,  // [num_blocks, num_heads, head_size, block_size]
           int64_t num_kv_heads, // [num_heads]
           double scale,
           Tensor &block_tables, // [num_seqs, max_num_blocks_per_seq]
           Tensor &seq_lens,     // [num_seqs]
           int64_t block_size,
           int64_t max_seq_len,
           const std::optional<Tensor> &alibi_slopes,
           const std::string &kv_cache_dtype,
           Tensor &k_scale,
           Tensor &v_scale,
           const int64_t tp_rank,
           const int64_t blocksparse_local_blocks,
           const int64_t blocksparse_vert_stride,
           const int64_t blocksparse_block_size,
           const int64_t blocksparse_head_sliding_step) {

    size_t seed = hash_combine(out, query, key_cache, value_cache, block_tables, seq_lens, alibi_slopes);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, PagedAttentionV1,
        seed,
        out->desc(), query->desc(), key_cache->desc(), value_cache->desc(),
        block_tables->desc(), seq_lens->desc(),
        alibi_slopes ? alibi_slopes.value()->desc() : nullptr,
        scale);

    INFINIOP_WORKSPACE_TENSOR(workspace, PagedAttentionV1, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(out),
        graph::GraphTensor(query),
        graph::GraphTensor(key_cache),
        graph::GraphTensor(value_cache),
        graph::GraphTensor(block_tables),
        graph::GraphTensor(seq_lens),
        alibi_slopes ? std::optional<graph::GraphTensor>(graph::GraphTensor(*alibi_slopes)) : std::nullopt,
        graph::GraphTensor(k_scale),
        graph::GraphTensor(v_scale),
        num_kv_heads,
        scale,
        block_size,
        max_seq_len,
        kv_cache_dtype,
        tp_rank,
        blocksparse_local_blocks,
        blocksparse_vert_stride,
        blocksparse_block_size,
        blocksparse_head_sliding_step};
}

void run(void *planned_meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(
        infiniopPagedAttentionV1(
            p->descriptor->desc,
            p->workspace->data(),
            p->workspace->numel(),
            p->out->data(),
            p->query->data(),
            p->key_cache->data(),
            p->value_cache->data(),
            p->num_kv_heads,
            p->scale,
            p->block_tables->data(),
            p->seq_lens->data(),
            p->block_size,
            p->max_seq_len,
            p->alibi_slopes.has_value() ? p->alibi_slopes.value()->data() : nullptr,
            p->kv_cache_dtype.c_str(),
            p->k_scale->data(),
            p->v_scale->data(),
            p->tp_rank,
            p->blocksparse_local_blocks,
            p->blocksparse_vert_stride,
            p->blocksparse_block_size,
            p->blocksparse_head_sliding_step,
            context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(PagedAttentionV1, &plan, &run, &cleanup);

} // namespace infinicore::op::paged_attention_v1_impl::infiniop
