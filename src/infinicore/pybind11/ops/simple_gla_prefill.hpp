#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/simple_gla_prefill.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline Tensor py_simple_gla_prefill(Tensor q,
                                    Tensor k,
                                    Tensor v,
                                    Tensor g_gamma,
                                    float scale) {
    return op::simple_gla_prefill(q, k, v, g_gamma, scale);
}

inline void bind_simple_gla_prefill(py::module &m) {
    m.def(
        "simple_gla_prefill",
        &ops::py_simple_gla_prefill,
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("g_gamma"),
        py::arg("scale"),
        R"doc(Simple GLA prefill fused kernel. q, k, v [B, T, H, D], g_gamma [H] (F32). Returns [B, T, H, D].)doc");
}

} // namespace infinicore::ops
