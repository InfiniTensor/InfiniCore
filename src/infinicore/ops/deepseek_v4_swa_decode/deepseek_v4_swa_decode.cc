#include "infinicore/ops/deepseek_v4_swa_decode.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4SwaDecode);

DeepseekV4SwaDecode::DeepseekV4SwaDecode(Tensor y,
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
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, q, k, attn_sink, positions);
    INFINICORE_GRAPH_OP_DISPATCH(
        y->device().getType(), y, q, k, attn_sink, positions, key_offset, key_len,
        softmax_scale, rope_dim, rope_theta, use_yarn, yarn_factor, yarn_beta_fast,
        yarn_beta_slow, yarn_original_seq_len, yarn_extrapolation_factor);
}

void DeepseekV4SwaDecode::execute(Tensor y,
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
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        DeepseekV4SwaDecode, y, q, k, attn_sink, positions, key_offset, key_len,
        softmax_scale, rope_dim, rope_theta, use_yarn, yarn_factor, yarn_beta_fast,
        yarn_beta_slow, yarn_original_seq_len, yarn_extrapolation_factor);
}

Tensor deepseek_v4_swa_decode(const Tensor &q,
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
    const auto &q_shape = q->shape();
    INFINICORE_ASSERT(q_shape.size() == 4);
    auto y = Tensor::empty(q_shape, q->dtype(), q->device());
    deepseek_v4_swa_decode_(y, q, k, attn_sink, positions, key_offset, key_len,
                            softmax_scale, rope_dim, rope_theta, use_yarn, yarn_factor,
                            yarn_beta_fast, yarn_beta_slow,
                            yarn_original_seq_len, yarn_extrapolation_factor);
    return y;
}

void deepseek_v4_swa_decode_(Tensor y,
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
    DeepseekV4SwaDecode::execute(y, q, k, attn_sink, positions, key_offset, key_len,
                                 softmax_scale, rope_dim, rope_theta, use_yarn, yarn_factor,
                                 yarn_beta_fast, yarn_beta_slow,
                                 yarn_original_seq_len, yarn_extrapolation_factor);
}

} // namespace infinicore::op
