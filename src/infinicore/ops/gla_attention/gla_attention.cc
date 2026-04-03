#include "infinicore/ops/gla_attention.hpp"

#include "infinicore/ops/matmul.hpp"
#include "infinicore/ops/causal_softmax.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

Tensor gla_attention(const Tensor &q,
                     const Tensor &k_total,
                     const Tensor &v_total,
                     float scale,
                     bool causal) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(q, k_total, v_total);

    const auto &q_shape = q->shape();       // [B, n_q, S_q, D]
    const auto &k_shape = k_total->shape(); // [B, n_kv, S_kv, D]
    const auto &v_shape = v_total->shape(); // [B, n_kv, S_kv, D]

    INFINICORE_ASSERT(q_shape.size() == 4);
    INFINICORE_ASSERT(k_shape.size() == 4);
    INFINICORE_ASSERT(v_shape.size() == 4);
    INFINICORE_ASSERT(q_shape[0] == k_shape[0] && k_shape[0] == v_shape[0]); // B
    INFINICORE_ASSERT(q_shape[3] == k_shape[3] && k_shape[3] == v_shape[3]); // D
    INFINICORE_ASSERT(k_shape[1] == v_shape[1] && k_shape[2] == v_shape[2]); // n_kv, S_kv

    const size_t B = q_shape[0];
    const size_t n_q = q_shape[1];
    const size_t S_q = q_shape[2];
    const size_t D = q_shape[3];
    const size_t n_kv = k_shape[1];
    const size_t S_kv = k_shape[2];

    INFINICORE_ASSERT(n_q % n_kv == 0);
    const size_t ngroup = n_q / n_kv;

    // Reshape to grouped GQA layout:
    //   Q: [B * n_kv, ngroup * S_q, D]
    //   K: [B * n_kv, S_kv, D]
    //   V: [B * n_kv, S_kv, D]
    auto Q = q->view({B * n_kv, ngroup, S_q, D})
                 ->view({B * n_kv, ngroup * S_q, D});
    auto K = k_total->view({B * n_kv, S_kv, D});
    auto V = v_total->view({B * n_kv, S_kv, D});

    auto Kt = K->permute({0, 2, 1}); // [B * n_kv, D, S_kv]
    auto attn_weight = infinicore::op::matmul(Q, Kt, scale); // [B * n_kv, ngroup * S_q, S_kv]

    if (causal) {
        auto attn_weight_softmax =
            attn_weight->view({B * n_q, S_q, S_kv}); // [B * n_q, S_q, S_kv]
        infinicore::op::causal_softmax_(attn_weight_softmax, attn_weight_softmax);
    }

    auto out = infinicore::op::matmul(attn_weight, V); // [B * n_kv, ngroup * S_q, D]
    auto out_view =
        out->view({B, n_kv, ngroup, S_q, D})
            ->view({B, n_q, S_q, D}); // merge kv,group back into n_q

    return out_view;
}

} // namespace infinicore::op

