#include "infinicore/ops/lightning_attention.hpp"
#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(LightningAttention);

LightningAttention::LightningAttention(Tensor out,
                                       Tensor initial_state,
                                       const Tensor &q,
                                       const Tensor &k,
                                       const Tensor &v,
                                       const Tensor &slope,
                                       const Tensor &initial_state_indices,
                                       const Tensor &final_state_indices) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, initial_state, q, k, v, slope);
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, initial_state_indices);
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, final_state_indices);
    INFINICORE_GRAPH_OP_DISPATCH(out->device().getType(),
                                 out,
                                 initial_state,
                                 q,
                                 k,
                                 v,
                                 slope,
                                 initial_state_indices,
                                 final_state_indices);
}

void LightningAttention::execute(Tensor out,
                                 Tensor initial_state,
                                 const Tensor &q,
                                 const Tensor &k,
                                 const Tensor &v,
                                 const Tensor &slope,
                                 const Tensor &initial_state_indices,
                                 const Tensor &final_state_indices) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(LightningAttention,
                                      out,
                                      initial_state,
                                      q,
                                      k,
                                      v,
                                      slope,
                                      initial_state_indices,
                                      final_state_indices);
}

static Tensor ensure_4d_sequence_tensor(const Tensor &x, const char *name) {
    if (x->shape().size() == 4) {
        return x;
    }
    if (x->shape().size() == 3) {
        return x->unsqueeze(1);
    }
    throw std::runtime_error(std::string("lightning_attention expects ") + name + " with shape [B, T, H, D] or [B, H, D]");
}

Tensor lightning_attention(const Tensor &q,
                           const Tensor &k,
                           const Tensor &v,
                           const Tensor &slope,
                           Tensor initial_state,
                           const Tensor &initial_state_indices,
                           const Tensor &final_state_indices) {
    Tensor q4 = ensure_4d_sequence_tensor(q, "q");
    Tensor k4 = ensure_4d_sequence_tensor(k, "k");
    Tensor v4 = ensure_4d_sequence_tensor(v, "v");
    auto out = Tensor::empty(v4->shape(), v4->dtype(), v4->device());
    lightning_attention_(out, initial_state, q4, k4, v4, slope, initial_state_indices, final_state_indices);
    return out;
}

void lightning_attention_(Tensor out,
                          Tensor initial_state,
                          const Tensor &q,
                          const Tensor &k,
                          const Tensor &v,
                          const Tensor &slope,
                          const Tensor &initial_state_indices,
                          const Tensor &final_state_indices) {
    Tensor q4 = ensure_4d_sequence_tensor(q, "q");
    Tensor k4 = ensure_4d_sequence_tensor(k, "k");
    Tensor v4 = ensure_4d_sequence_tensor(v, "v");
    LightningAttention::execute(out, initial_state, q4, k4, v4, slope, initial_state_indices, final_state_indices);
}

} // namespace infinicore::op
