#pragma once

#include "infinicore/ops/linear.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_linear(py::module &m) {
    m.def("linear",
          &op::linear,
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          R"doc(Applies a linear transformation to the incoming data: y=xA^T+b.)doc");

    m.def("linear_",
          &op::linear_,
          py::arg("out"),
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          R"doc(In-place, applies a linear transformation to the incoming data: y=xA^T+b.)doc");
}

} // namespace infinicore::ops
