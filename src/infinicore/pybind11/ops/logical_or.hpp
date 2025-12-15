#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/logical_or.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_logical_or(py::module &m) {
    m.def("logical_or",
          &op::logical_or,
          py::arg("a"),
          py::arg("b"),
          R"doc(Logical OR of two tensors.)doc");

    m.def("logical_or_",
          &op::logical_or_,
          py::arg("c"),
          py::arg("a"),
          py::arg("b"),
          R"doc(In-place logical OR of two tensors.)doc");
}

} // namespace infinicore::ops

