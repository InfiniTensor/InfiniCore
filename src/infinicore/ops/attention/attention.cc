#include "infinicore/ops/attention.hpp"
#include "infinicore/ops/causal_softmax.hpp"
#include "infinicore/ops/gemm.hpp"

namespace infinicore::op {

common::OpDispatcher<Attention::schema> &Attention::dispatcher() {
    static common::OpDispatcher<Attention::schema> dispatcher_;
    return dispatcher_;
};

void Attention::execute(Tensor out, Tensor q, Tensor k, Tensor v, Tensor k_cache, Tensor v_cache, size_t pos) {
    dispatcher().lookup(context::getDevice().getType())(out, q, k, v, k_cache, v_cache, pos);
}

Tensor attention(Tensor q, Tensor k, Tensor v, Tensor k_cache, Tensor v_cache, size_t pos) {
    size_t n_q_head = q->shape()[0];
    size_t seq_len = q->shape()[1];
    size_t head_dim = q->shape()[2];
    Shape shape = {seq_len, n_q_head, head_dim};
    auto out = Tensor::empty(shape, q->dtype(), q->device());
    attention_(out, q, k, v, k_cache, v_cache, pos);
    return out;
}

void attention_(Tensor out, Tensor q, Tensor k, Tensor v, Tensor k_cache, Tensor v_cache, size_t pos) {
    Attention::execute(out, q, k, v, k_cache, v_cache, pos);
}

Tensor scaled_dot_product_attention(Tensor query_states, // [bs, num_attention_heads, ntoken, head_dim]
                                    Tensor key_states,   // [bs, num_key_value_heads, total_token, head_dim]
                                    Tensor value_states, // [bs, num_key_value_heads, total_token, head_dim]
                                    pybind11::object scale) {

    auto query_shape = query_states->shape();
    auto key_shape = key_states->shape();

    Size batch_size = query_shape[0];
    Size num_attention_heads = query_shape[1];
    Size ntoken = query_shape[2];
    Size head_dim = key_shape[3];

    Tensor output_values = Tensor::empty({batch_size, num_attention_heads, ntoken, head_dim}, query_states->dtype(), query_states->device());

    scaled_dot_product_attention_(output_values, query_states, key_states, value_states, scale);

    return output_values;
}

void scaled_dot_product_attention_(Tensor out,
                                   Tensor query_states,
                                   Tensor key_states,
                                   Tensor value_states,
                                   pybind11::object scale) {

    auto query_shape = query_states->shape();
    auto key_shape = key_states->shape();

    Size batch_size = query_shape[0];
    Size num_attention_heads = query_shape[1];
    Size ntoken = query_shape[2];

    Size num_key_value_heads = key_shape[1];
    Size total_token = key_shape[2];
    Size head_dim = key_shape[3];

    assert(0 == (num_attention_heads % num_key_value_heads));
    Size ngroup = num_attention_heads / num_key_value_heads;

    float attention_scale{0.0f};
    if (!scale.is_none()) {
        attention_scale = scale.cast<float>();
    } else {
        attention_scale = 1.f / float(sqrt(head_dim));
    }

    Tensor out_view = out->view({batch_size, num_key_value_heads, ngroup * ntoken, head_dim});
    for (Size ib = 0; ib < batch_size; ++ib) {
        Tensor q = query_states->narrow({{0, ib, 1}})->view({num_attention_heads, ntoken, head_dim});      // [ num_attention_heads, ntoken, head_dim]
        Tensor k = key_states->narrow({{0, ib, 1}})->view({num_key_value_heads, total_token, head_dim});   // [ num_key_value_heads, total_token, head_dim]
        Tensor v = value_states->narrow({{0, ib, 1}})->view({num_key_value_heads, total_token, head_dim}); // [ num_key_value_heads, total_token, head_dim]
        Tensor output_v = out_view->narrow({{0, ib, 1}})->view({num_key_value_heads, ngroup * ntoken, head_dim});
        {
            /*
            输入：
                q,  [ num_attention_heads, ntoken, head_dim]
                k,  [ num_key_value_heads, total_token, head_dim]
                v,  [ num_key_value_heads, total_token, head_dim]
            输出：
                att_val ： {num_key_value_heads, ngroup * ntok, head_dim}
            */

            auto q_gemm = q->view({num_key_value_heads, ngroup * ntoken, head_dim}); // => {nkvh, ngroup * seq_len, dh}
            auto k_gemm = k->permute({0, 2, 1});                                     // => { nkvh, dh, total_token}
            auto v_gemm = v;                                                         // => { nkvh, total_token, dh}

            // qk_score : => {nkvh, ngroup * ntoken, total_token}
            Tensor qk_score = gemm(q_gemm, // {nkvh, ngroup * ntoken, dh}
                                   k_gemm, // {nkvh, dh, total_token}
                                   attention_scale, 0.f);

            // softmax

            auto qk_softmax = qk_score->view({num_attention_heads, ntoken, total_token});
            causal_softmax_(qk_softmax, qk_softmax);

            // values
            gemm_(output_v, // {nkvh, ngroup * ntoken, dh}
                  qk_score, // {nkvh, ngroup * ntoken, total_token}
                  v_gemm,   // { nkvh, total_token, dh}
                  1.0f, 0.0f);
        }
    }
}
} // namespace infinicore::op
