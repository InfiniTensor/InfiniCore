#pragma once

#include "infinicore/ops/binary_cross_entropy.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace infinicore::ops {

inline void bind_binary_cross_entropy(py::module &m) {
    m.def("binary_cross_entropy",
          &op::binary_cross_entropy,
          py::arg("input"),
          py::arg("target"),
          py::arg("weight") = py::none(),
          py::arg("reduction") = "mean",
          R"doc(Calculates Binary Cross Entropy.)doc");

    m.def("binary_cross_entropy_",
          &op::binary_cross_entropy_,
          py::arg("input"),
          py::arg("target"),
          py::arg("weight"),
          py::arg("output"),
          py::arg("reduction") = "mean",
          R"doc(In-place Binary Cross Entropy calculation.)doc");
}

} // namespace infinicore::ops