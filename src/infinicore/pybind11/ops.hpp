#pragma once

#include <pybind11/pybind11.h>

#include "ops/add.hpp"
#include "ops/attention.hpp"
#include "ops/cat.hpp"
#include "ops/causal_softmax.hpp"
#include "ops/embedding.hpp"
#include "ops/inner.hpp"
#include "ops/linear.hpp"
#include "ops/masked_select.hpp"
#include "ops/matmul.hpp"
#include "ops/mul.hpp"
#include "ops/random_sample.hpp"
#include "ops/rearrange.hpp"
#include "ops/rms_norm.hpp"
#include "ops/rope.hpp"
#include "ops/silu.hpp"
#include "ops/swiglu.hpp"
#include "ops/tan.hpp"
#include "ops/tanhshrink.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind(py::module &m) {
    bind_add(m);
    bind_attention(m);
    bind_cat(m);
    bind_causal_softmax(m);
    bind_inner(m);
    bind_random_sample(m);
    bind_linear(m);
    bind_masked_select(m);
    bind_matmul(m);
    bind_mul(m);
    bind_rearrange(m);
    bind_rms_norm(m);
    bind_silu(m);
    bind_swiglu(m);
    bind_tan(m);
    bind_tanhshrink(m);
    bind_rope(m);
    bind_embedding(m);
}

} // namespace infinicore::ops
