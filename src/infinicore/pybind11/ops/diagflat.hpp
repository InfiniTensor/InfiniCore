#pragma once

#include "infinicore/ops/diagflat.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_diagflat(py::module &m) {
    m.def(
        "diagflat",
        &op::diagflat,
        py::arg("input"),
        py::arg("offset") = 0,
        R"doc(Create a 2D matrix with the input flattened into the diagonal.)doc");
    m.def(
        "diagflat_",
        &op::diagflat_,
        py::arg("output"),
        py::arg("input"),
        py::arg("offset") = 0,
        R"doc(In-place diagflat into the given output tensor.)doc");
}

} // namespace infinicore::ops


