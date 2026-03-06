#pragma once

#include <pybind11/pybind11.h>

#include "ops/add.hpp"
#include "ops/add_rms_norm.hpp"
#include "ops/attention.hpp"
#include "ops/bitwise_right_shift.hpp"
#include "ops/causal_softmax.hpp"
#include "ops/embedding.hpp"
#include "ops/flash_attention.hpp"
#include "ops/gaussian_nll_loss.hpp"
#include "ops/interpolate.hpp"
#include "ops/kv_caching.hpp"
#include "ops/linear.hpp"
#include "ops/linear_w8a8i8.hpp"
#include "ops/matmul.hpp"
#include "ops/mul.hpp"
#include "ops/paged_attention.hpp"
#include "ops/paged_attention_prefill.hpp"
#include "ops/paged_caching.hpp"
#include "ops/prelu.hpp"
#include "ops/random_sample.hpp"
#include "ops/rearrange.hpp"
#include "ops/relu6.hpp"
#include "ops/rms_norm.hpp"
#include "ops/rope.hpp"
#include "ops/silu.hpp"
#include "ops/silu_and_mul.hpp"
#include "ops/swiglu.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind(py::module &m) {
    bind_add(m);
    bind_add_rms_norm(m);
    bind_attention(m);
    bind_bitwise_right_shift(m);
    bind_causal_softmax(m);
    bind_flash_attention(m);
    bind_kv_caching(m);
    bind_linear(m);
    bind_matmul(m);
    bind_mul(m);
    bind_gaussian_nll_loss(m);
    bind_interpolate(m);
    bind_paged_attention(m);
    bind_paged_attention_prefill(m);
    bind_paged_caching(m);
    bind_prelu(m);
    bind_random_sample(m);
    bind_rearrange(m);
    bind_relu6(m);
    bind_rms_norm(m);
    bind_silu(m);
    bind_swiglu(m);
    bind_rope(m);
    bind_embedding(m);
    bind_linear_w8a8i8(m);
    bind_silu_and_mul(m);
}

} // namespace infinicore::ops
