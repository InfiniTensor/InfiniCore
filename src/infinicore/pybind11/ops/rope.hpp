#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/rope.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_rope(py::module &m) {
    m.def("rope",
          &op::rope,
          py::arg("x"),
          py::arg("pos"),
          py::arg("sin_table"),
          py::arg("cos_table"),
          py::arg("algo"),
          R"doc( Rotary Position Embedding(RoPE).)doc");

    m.def("rope_",
          &op::rope_,
          py::arg("x_out"),
          py::arg("x"),
          py::arg("pos"),
          py::arg("sin_table"),
          py::arg("cos_table"),
          py::arg("algo"),
          R"doc(In-place, Rotary Position Embedding(RoPE).)doc");
}

} // namespace infinicore::ops
