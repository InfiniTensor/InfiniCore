#include "infinicore/ops/kimi_delta_attention.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(KimiDeltaAttention);

KimiDeltaAttention::KimiDeltaAttention(Tensor out,
                                       Tensor initial_state,
                                       std::optional<Tensor> final_state,
                                       const Tensor &q,
                                       const Tensor &k,
                                       const Tensor &v,
                                       const Tensor &g,
                                       const Tensor &beta,
                                       const Tensor &A_log,
                                       const Tensor &dt_bias,
                                       std::optional<Tensor> cu_seqlens,
                                       std::optional<Tensor> initial_state_indices,
                                       std::optional<Tensor> final_state_indices,
                                       float scale,
                                       float lower_bound,
                                       bool use_qk_l2norm) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, initial_state, q, k, v, g, beta, A_log, dt_bias);
    if (final_state.has_value()) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, final_state.value());
    }
    if (cu_seqlens.has_value()) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, cu_seqlens.value());
    }
    if (initial_state_indices.has_value()) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, initial_state_indices.value());
    }
    if (final_state_indices.has_value()) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, final_state_indices.value());
    }
    INFINICORE_GRAPH_OP_DISPATCH(out->device().getType(),
                                 out,
                                 initial_state,
                                 final_state,
                                 q,
                                 k,
                                 v,
                                 g,
                                 beta,
                                 A_log,
                                 dt_bias,
                                 cu_seqlens,
                                 initial_state_indices,
                                 final_state_indices,
                                 scale,
                                 lower_bound,
                                 use_qk_l2norm);
}

void KimiDeltaAttention::execute(Tensor out,
                                 Tensor initial_state,
                                 std::optional<Tensor> final_state,
                                 const Tensor &q,
                                 const Tensor &k,
                                 const Tensor &v,
                                 const Tensor &g,
                                 const Tensor &beta,
                                 const Tensor &A_log,
                                 const Tensor &dt_bias,
                                 std::optional<Tensor> cu_seqlens,
                                 std::optional<Tensor> initial_state_indices,
                                 std::optional<Tensor> final_state_indices,
                                 float scale,
                                 float lower_bound,
                                 bool use_qk_l2norm) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(KimiDeltaAttention,
                                      out,
                                      initial_state,
                                      final_state,
                                      q,
                                      k,
                                      v,
                                      g,
                                      beta,
                                      A_log,
                                      dt_bias,
                                      cu_seqlens,
                                      initial_state_indices,
                                      final_state_indices,
                                      scale,
                                      lower_bound,
                                      use_qk_l2norm);
}

static void check_inputs(const Tensor &q,
                         const Tensor &k,
                         const Tensor &v,
                         const Tensor &g,
                         const Tensor &beta,
                         const Tensor &A_log,
                         const Tensor &dt_bias) {
    if (q->shape().size() != 4 || k->shape() != q->shape() || v->shape() != q->shape() || g->shape() != q->shape()) {
        throw std::runtime_error("kimi_delta_attention expects q/k/v/g with shape [B, T, H, D] or [1, total_tokens, H, D]");
    }
    if (beta->shape().size() != 3 || beta->shape()[0] != q->shape()[0] || beta->shape()[1] != q->shape()[1] || beta->shape()[2] != q->shape()[2]) {
        throw std::runtime_error("kimi_delta_attention expects beta with shape [B, T, H] or [1, total_tokens, H]");
    }
    if (A_log->shape().size() != 1 || A_log->shape()[0] != q->shape()[2] || dt_bias->shape().size() != 2 || dt_bias->shape()[0] != q->shape()[2] || dt_bias->shape()[1] != q->shape()[3]) {
        throw std::runtime_error("kimi_delta_attention expects A_log [H] and dt_bias [H, D]");
    }
}

static Shape final_state_shape(const Tensor &q, std::optional<Tensor> cu_seqlens) {
    size_t B = cu_seqlens.has_value() ? cu_seqlens.value()->shape()[0] - 1 : q->shape()[0];
    return {B, q->shape()[2], q->shape()[3], q->shape()[3]};
}

Tensor kimi_delta_attention(const Tensor &q,
                            const Tensor &k,
                            const Tensor &v,
                            const Tensor &g,
                            const Tensor &beta,
                            const Tensor &A_log,
                            const Tensor &dt_bias,
                            Tensor initial_state,
                            std::optional<Tensor> cu_seqlens,
                            std::optional<Tensor> initial_state_indices,
                            std::optional<Tensor> final_state_indices,
                            float scale,
                            float lower_bound,
                            bool use_qk_l2norm) {
    check_inputs(q, k, v, g, beta, A_log, dt_bias);
    Tensor out = Tensor::empty(v->shape(), v->dtype(), v->device());
    std::optional<Tensor> final_state = std::nullopt;
    if (!final_state_indices.has_value()) {
        final_state = Tensor::empty(final_state_shape(q, cu_seqlens), initial_state->dtype(), initial_state->device());
    }
    kimi_delta_attention_(out,
                          initial_state,
                          final_state,
                          q,
                          k,
                          v,
                          g,
                          beta,
                          A_log,
                          dt_bias,
                          cu_seqlens,
                          initial_state_indices,
                          final_state_indices,
                          scale,
                          lower_bound,
                          use_qk_l2norm);
    return out;
}

void kimi_delta_attention_(Tensor out,
                           Tensor initial_state,
                           std::optional<Tensor> final_state,
                           const Tensor &q,
                           const Tensor &k,
                           const Tensor &v,
                           const Tensor &g,
                           const Tensor &beta,
                           const Tensor &A_log,
                           const Tensor &dt_bias,
                           std::optional<Tensor> cu_seqlens,
                           std::optional<Tensor> initial_state_indices,
                           std::optional<Tensor> final_state_indices,
                           float scale,
                           float lower_bound,
                           bool use_qk_l2norm) {
    check_inputs(q, k, v, g, beta, A_log, dt_bias);
    if (out->shape() != v->shape()) {
        throw std::runtime_error("kimi_delta_attention_ output shape must match v");
    }
    KimiDeltaAttention::execute(out,
                                initial_state,
                                final_state,
                                q,
                                k,
                                v,
                                g,
                                beta,
                                A_log,
                                dt_bias,
                                cu_seqlens,
                                initial_state_indices,
                                final_state_indices,
                                scale,
                                lower_bound,
                                use_qk_l2norm);
}

} // namespace infinicore::op
