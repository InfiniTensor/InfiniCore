#pragma once

#include <pybind11/pybind11.h>
#include "ops/acos.hpp"
#include "ops/adaptive_avg_pool1d.hpp"
#include "ops/addbmm.hpp"
#include "ops/add.hpp"
#include "ops/affine_grid.hpp" 
#include "ops/attention.hpp"
#include "ops/causal_softmax.hpp"
#include "ops/embedding.hpp"
#include "ops/floor.hpp"
#include "ops/hypot.hpp"
#include "ops/index_add.hpp"
#include "ops/index_copy.hpp"
#include "ops/linear.hpp"
#include "ops/matmul.hpp"
#include "ops/mul.hpp"
#include "ops/random_sample.hpp"
#include "ops/rearrange.hpp"
#include "ops/rms_norm.hpp"
#include "ops/rope.hpp"
#include "ops/silu.hpp"
#include "ops/smooth_l1_loss.hpp"
#include "ops/swiglu.hpp"
#include "ops/take.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind(py::module &m) {
    bind_acos(m);
    bind_adaptive_avg_pool1d(m);
    bind_add(m);
    bind_affine_grid(m); 
    bind_addbmm(m);
    bind_attention(m);
    bind_causal_softmax(m);
    bind_random_sample(m);
    bind_floor(m);
    bind_hypot(m);
    bind_index_add(m);
    bind_index_copy(m);
    bind_linear(m);
    bind_matmul(m);
    bind_mul(m);
    bind_rearrange(m);
    bind_rms_norm(m);
    bind_silu(m);
    bind_smooth_l1_loss(m);
    bind_swiglu(m);
    bind_take(m);
    bind_rope(m);
    bind_embedding(m);
}

} // namespace infinicore::ops