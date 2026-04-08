#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/simple_gla_attention.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline Tensor py_simple_gla_attention(Tensor q,
                                      Tensor k,
                                      Tensor v,
                                      Tensor g_gamma,
                                      float scale) {
    return op::simple_gla_attention(q, k, v, g_gamma, scale);
}

inline void bind_simple_gla_attention(py::module &m) {
    m.def(
        "simple_gla_attention",
        &ops::py_simple_gla_attention,
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("g_gamma"),
        py::arg("scale"),
        R"doc(Simple GLA (recurrent linear) attention. q, k, v [B, T, H, D], g_gamma [H]. Returns [B, T, H, D].)doc");
}

} // namespace infinicore::ops
