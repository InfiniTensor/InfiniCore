#pragma once

#include <pybind11/pybind11.h>

#include "ops/add.hpp"
#include "ops/add_rms_norm.hpp"
#include "ops/addcmul.hpp"
#include "ops/all.hpp"
#include "ops/atanh.hpp"
#include "ops/attention.hpp"
#include "ops/avg_pool1d.hpp"
#include "ops/binary_cross_entropy_with_logits.hpp"
#include "ops/causal_softmax.hpp"
#include "ops/cdist.hpp"
#include "ops/cross_entropy.hpp"
#include "ops/embedding.hpp"
#include "ops/equal.hpp"
#include "ops/flash_attention.hpp"
#include "ops/gla_attention.hpp"
#include "ops/infllmv2_attention.hpp"
#include "ops/simple_gla_attention.hpp"
#include "ops/simple_gla_decode_step.hpp"
#include "ops/simple_gla_prefill.hpp"
#include "ops/hardswish.hpp"
#include "ops/hardtanh.hpp"
#include "ops/kv_caching.hpp"
#include "ops/linear.hpp"
#include "ops/linear_w8a8i8.hpp"
#include "ops/matmul.hpp"
#include "ops/mha_kvcache.hpp"
#include "ops/mha_varlen.hpp"
#include "ops/mul.hpp"
#include "ops/paged_attention.hpp"
#include "ops/paged_attention_prefill.hpp"
#include "ops/paged_caching.hpp"
#include "ops/random_sample.hpp"
#include "ops/rearrange.hpp"
#include "ops/reciprocal.hpp"
#include "ops/rms_norm.hpp"
#include "ops/rope.hpp"
#include "ops/silu.hpp"
#include "ops/silu_and_mul.hpp"
#include "ops/sum.hpp"
#include "ops/swiglu.hpp"
#include "ops/topk.hpp"
#include "ops/var.hpp"
#include "ops/var_mean.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind(py::module &m) {
    bind_add(m);
    bind_add_rms_norm(m);
    bind_attention(m);
    bind_causal_softmax(m);
    bind_flash_attention(m);
    bind_gla_attention(m);
    bind_infllmv2_attention(m);
    bind_simple_gla_attention(m);
    bind_simple_gla_decode_step(m);
    bind_simple_gla_prefill(m);
    bind_kv_caching(m);
    bind_linear(m);
    bind_matmul(m);
    bind_mul(m);
    bind_mha_kvcache(m);
    bind_mha_varlen(m);
    bind_hardswish(m);
    bind_hardtanh(m);
    bind_paged_attention(m);
    bind_paged_attention_prefill(m);
    bind_paged_caching(m);
    bind_random_sample(m);
    bind_cross_entropy(m);
    bind_rearrange(m);
    bind_rms_norm(m);
    bind_avg_pool1d(m);
    bind_silu(m);
    bind_swiglu(m);
    bind_rope(m);
    bind_embedding(m);
    bind_linear_w8a8i8(m);
    bind_silu_and_mul(m);
    bind_sum(m);
    bind_var_mean(m);
    bind_var(m);
    bind_topk(m);
    bind_all(m);
    bind_equal(m);
    bind_atanh(m);
    bind_addcmul(m);
    bind_cdist(m);
    bind_binary_cross_entropy_with_logits(m);
    bind_reciprocal(m);
}

} // namespace infinicore::ops
