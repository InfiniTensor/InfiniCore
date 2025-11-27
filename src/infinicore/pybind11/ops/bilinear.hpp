#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/bilinear.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_bilinear(py::module &m) {
    m.def("bilinear",
          &op::bilinear,
          py::arg("x1"),
          py::arg("x2"),
          py::arg("weight"),
          py::arg("bias"),
          R"doc(Bilinear transformation of two input tensors.)doc");
    m.def("bilinear_",
          &op::bilinear_,
          py::arg("out"),
          py::arg("x1"),
          py::arg("x2"),
          py::arg("weight"),
          py::arg("bias"),
          R"doc(In-place bilinear transformation of two input tensors.)doc");
}   

} // namespace infinicore::ops  