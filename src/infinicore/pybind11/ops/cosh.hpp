#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/cosh.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_cosh(py::module &m) {
    m.def("cosh",
          &op::cosh,
          py::arg("x"),
          R"doc(Element-wise hyperbolic cosine function.)doc");

    m.def("cosh_",
          &op::cosh_,
          py::arg("y"),
          py::arg("x"),
          R"doc(In-place element-wise hyperbolic cosine function.)doc");
}

} // namespace infinicore::ops