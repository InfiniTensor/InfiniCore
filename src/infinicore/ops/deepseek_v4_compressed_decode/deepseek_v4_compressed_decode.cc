#include "infinicore/ops/deepseek_v4_compressed_decode.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4CompressedDecode);

DeepseekV4CompressedDecode::DeepseekV4CompressedDecode(
    Tensor y,
    const Tensor &q,
    const Tensor &k,
    const Tensor &kv_comp,
    const Tensor &attn_sink,
    const Tensor &query_positions,
    const Tensor &block_positions,
    const Tensor &indexed_blocks,
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
    if (indexed_blocks) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, q, k, kv_comp, attn_sink, query_positions, block_positions, indexed_blocks);
    } else {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, q, k, kv_comp, attn_sink, query_positions, block_positions);
    }
    INFINICORE_GRAPH_OP_DISPATCH(y->device().getType(), y, q, k, kv_comp, attn_sink,
                                 query_positions, block_positions, indexed_blocks,
                                 softmax_scale, compress_ratio, index_top_k, rope_dim,
                                 rope_theta, use_yarn, yarn_factor, yarn_beta_fast,
                                 yarn_beta_slow, yarn_original_seq_len,
                                 yarn_extrapolation_factor);
}

void DeepseekV4CompressedDecode::execute(
    Tensor y,
    const Tensor &q,
    const Tensor &k,
    const Tensor &kv_comp,
    const Tensor &attn_sink,
    const Tensor &query_positions,
    const Tensor &block_positions,
    const Tensor &indexed_blocks,
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
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(DeepseekV4CompressedDecode, y, q, k, kv_comp,
                                      attn_sink, query_positions, block_positions,
                                      indexed_blocks, softmax_scale, compress_ratio,
                                      index_top_k, rope_dim, rope_theta, use_yarn,
                                      yarn_factor, yarn_beta_fast, yarn_beta_slow,
                                      yarn_original_seq_len,
                                      yarn_extrapolation_factor);
}

Tensor deepseek_v4_compressed_decode(const Tensor &q,
                                     const Tensor &k,
                                     const Tensor &kv_comp,
                                     const Tensor &attn_sink,
                                     const Tensor &query_positions,
                                     const Tensor &block_positions,
                                     const Tensor &indexed_blocks,
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
    const auto &q_shape = q->shape();
    INFINICORE_ASSERT(q_shape.size() == 4);
    auto y = Tensor::empty(q_shape, q->dtype(), q->device());
    deepseek_v4_compressed_decode_(y, q, k, kv_comp, attn_sink, query_positions,
                                   block_positions, indexed_blocks, softmax_scale,
                                   compress_ratio, index_top_k, rope_dim, rope_theta,
                                   use_yarn, yarn_factor, yarn_beta_fast, yarn_beta_slow,
                                   yarn_original_seq_len, yarn_extrapolation_factor);
    return y;
}

void deepseek_v4_compressed_decode_(Tensor y,
                                    const Tensor &q,
                                    const Tensor &k,
                                    const Tensor &kv_comp,
                                    const Tensor &attn_sink,
                                    const Tensor &query_positions,
                                    const Tensor &block_positions,
                                    const Tensor &indexed_blocks,
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
    DeepseekV4CompressedDecode::execute(y, q, k, kv_comp, attn_sink, query_positions,
                                        block_positions, indexed_blocks, softmax_scale,
                                        compress_ratio, index_top_k, rope_dim,
                                        rope_theta, use_yarn, yarn_factor,
                                        yarn_beta_fast, yarn_beta_slow,
                                        yarn_original_seq_len, yarn_extrapolation_factor);
}

} // namespace infinicore::op
