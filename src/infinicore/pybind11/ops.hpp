#pragma once

#include <pybind11/pybind11.h>

#include "ops/add.hpp"
#include "ops/attention.hpp"
#include "ops/avg_pool3d.hpp"
#include "ops/causal_softmax.hpp"
#include "ops/dot.hpp"
#include "ops/embedding.hpp"
#include "ops/histc.hpp"
#include "ops/linear.hpp"
#include "ops/log10.hpp"
#include "ops/log1p.hpp"
#include "ops/matmul.hpp"
#include "ops/mul.hpp"
#include "ops/random_sample.hpp"
#include "ops/rearrange.hpp"
#include "ops/rms_norm.hpp"
#include "ops/rope.hpp"
#include "ops/silu.hpp"
#include "ops/swiglu.hpp"
#include "ops/zeros_.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind(py::module &m) {
    bind_add(m);
    bind_attention(m);
    bind_causal_softmax(m);
    bind_random_sample(m);
    bind_linear(m);
    bind_matmul(m);
    bind_mul(m);
    bind_rearrange(m);
    bind_rms_norm(m);
    bind_silu(m);
    bind_swiglu(m);
    bind_rope(m);
    bind_embedding(m);
    bind_histc(m);
    bind_zeros_(m);
    bind_log10(m);
    bind_avg_pool3d(m);
    bind_dot(m);
    bind_log1p(m);
}

} // namespace infinicore::ops
