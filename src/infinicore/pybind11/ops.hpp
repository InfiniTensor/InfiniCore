#pragma once

#include <pybind11/pybind11.h>

#include "ops/adaptive_max_pool1d.hpp"
#include "ops/add.hpp"
#include "ops/asinh.hpp"
#include "ops/attention.hpp"
#include "ops/baddbmm.hpp"
#include "ops/bilinear.hpp"
#include "ops/causal_softmax.hpp"
#include "ops/embedding.hpp"
#include "ops/fmod.hpp"
#include "ops/linear.hpp"
#include "ops/matmul.hpp"
#include "ops/mul.hpp"
#include "ops/random_sample.hpp"
#include "ops/rearrange.hpp"
#include "ops/rms_norm.hpp"
#include "ops/rope.hpp"
#include "ops/silu.hpp"
#include "ops/swiglu.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind(py::module &m) {
    bind_adaptive_max_pool1d(m);
    bind_add(m);
    bind_attention(m);
    bind_asinh(m);
    bind_baddbmm(m);
    bind_bilinear(m);
    bind_causal_softmax(m);
    bind_fmod(m);
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
}

} // namespace infinicore::ops
