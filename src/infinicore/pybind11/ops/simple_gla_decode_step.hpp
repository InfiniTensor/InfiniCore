#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/simple_gla_decode_step.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline Tensor py_simple_gla_decode_step(Tensor q, Tensor k, Tensor v, Tensor state, Tensor g_gamma, float scale) {
    return op::simple_gla_decode_step(q, k, v, state, g_gamma, scale);
}

inline void bind_simple_gla_decode_step(py::module &m) {
    m.def(
        "simple_gla_decode_step",
        &ops::py_simple_gla_decode_step,
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("state"),
        py::arg("g_gamma"),
        py::arg("scale"),
        R"doc(Simple GLA one decode step. q,k,v [B,1,H,D] (same dtype); state [B,H,D,D] float32 in-place; g_gamma [H]. Returns [B,1,H,D].)doc");
}

} // namespace infinicore::ops
