#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/where.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_where(py::module &m) {
    m.def("where",
          &op::where,
          py::arg("cond"),
          py::arg("x"),
          py::arg("y"),
          R"doc(Elementwise where(cond, x, y) selection.)doc");

    m.def("where_",
          &op::where_,
          py::arg("out"),
          py::arg("cond"),
          py::arg("x"),
          py::arg("y"),
          R"doc(In-place elementwise where(cond, x, y) selection into out tensor.)doc");

    m.def("where_indices",
          &op::where_indices,
          py::arg("cond"),
          R"doc(Return a tuple of index tensors where condition is True.)doc");
}

} // namespace infinicore::ops


