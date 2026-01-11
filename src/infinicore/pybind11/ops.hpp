#pragma once

#include <pybind11/pybind11.h>

#include "ops/add.hpp"
#include "ops/add_rms_norm.hpp"
#include "ops/attention.hpp"
#include "ops/causal_softmax.hpp"
#include "ops/embedding.hpp"
#include "ops/linear.hpp"
#include "ops/matmul.hpp"
#include "ops/mul.hpp"
#include "ops/paged_attention.hpp"
#include "ops/paged_caching.hpp"
#include "ops/random_sample.hpp"
#include "ops/rearrange.hpp"
#include "ops/rms_norm.hpp"
#include "ops/rope.hpp"
#include "ops/silu.hpp"
#include "ops/swiglu.hpp"
#include "ops/vander.hpp"
#include "ops/unfold.hpp"
#include "ops/logcumsumexp.hpp"
#include "ops/logical_and.hpp" 
#include "ops/logical_not.hpp" 
namespace py = pybind11;

namespace infinicore::ops {

inline void bind(py::module &m) {
    bind_add(m);
    bind_add_rms_norm(m);
    bind_attention(m);
    bind_causal_softmax(m);
    bind_random_sample(m);
    bind_linear(m);
    bind_matmul(m);
    bind_mul(m);
    bind_paged_attention(m);
    bind_paged_caching(m);
    bind_rearrange(m);
    bind_rms_norm(m);
    bind_silu(m);
    bind_swiglu(m);
    bind_logcumsumexp(m);
    bind_logical_and(m); 
    bind_logical_not(m);
    bind_vander(m);
    bind_unfold(m); 
    bind_rope(m);
    bind_embedding(m);
}

} // namespace infinicore::ops
