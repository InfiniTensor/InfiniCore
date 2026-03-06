#pragma once

#include <pybind11/pybind11.h>

#include "ops/adaptive_max_pool1d.hpp"
#include "ops/add.hpp"
#include "ops/add_rms_norm.hpp"
#include "ops/asinh.hpp"
#include "ops/attention.hpp"
#include "ops/baddbmm.hpp"
#include "ops/bilinear.hpp"
#include "ops/causal_softmax.hpp"
#include "ops/embedding.hpp"
#include "ops/flash_attention.hpp"
#include "ops/fmod.hpp"
#include "ops/kv_caching.hpp"
#include "ops/linear.hpp"
#include "ops/linear_w8a8i8.hpp"
#include "ops/matmul.hpp"
#include "ops/mha_varlen.hpp"
#include "ops/mul.hpp"
#include "ops/paged_attention.hpp"
#include "ops/paged_attention_prefill.hpp"
#include "ops/paged_caching.hpp"
#include "ops/random_sample.hpp"
#include "ops/rearrange.hpp"
#include "ops/rms_norm.hpp"
#include "ops/rope.hpp"
#include "ops/silu.hpp"
#include "ops/silu_and_mul.hpp"
#include "ops/swiglu.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind(py::module &m) {
    bind_adaptive_max_pool1d(m);
    bind_add(m);
    bind_add_rms_norm(m);
    bind_attention(m);
    bind_asinh(m);
    bind_baddbmm(m);
    bind_bilinear(m);
    bind_causal_softmax(m);
    bind_flash_attention(m);
    bind_kv_caching(m);
    bind_fmod(m);
    bind_random_sample(m);
    bind_linear(m);
    bind_matmul(m);
    bind_mul(m);
    bind_mha_varlen(m);
    bind_paged_attention(m);
    bind_paged_attention_prefill(m);
    bind_paged_caching(m);
    bind_random_sample(m);
    bind_rearrange(m);
    bind_rms_norm(m);
    bind_silu(m);
    bind_swiglu(m);
    bind_rope(m);
    bind_embedding(m);
    bind_linear_w8a8i8(m);
    bind_silu_and_mul(m);
}

} // namespace infinicore::ops
