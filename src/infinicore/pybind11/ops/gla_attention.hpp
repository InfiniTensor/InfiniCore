#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/gla_attention.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline Tensor py_gla_attention(Tensor q,
                               Tensor k_total,
                               Tensor v_total,
                               float scale,
                               bool causal) {
    return op::gla_attention(q, k_total, v_total, scale, causal);
}

inline void bind_gla_attention(py::module &m) {
    m.def(
        "gla_attention",
        &ops::py_gla_attention,
        py::arg("q"),
        py::arg("k_total"),
        py::arg("v_total"),
        py::arg("scale"),
        py::arg("causal") = true,
        R"doc(GLA-style attention: Q,K,V shapes [B, n_q/n_kv, S, D], scale, causal.
    Returns [B, n_q, S_q, D].)doc");
}

} // namespace infinicore::ops
