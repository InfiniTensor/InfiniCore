#pragma once

#include "infinicore/ops/minimum.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace infinicore::ops {

inline void bind_minimum(py::module &m) {
    m.def("minimum",
          &op::minimum,
          py::arg("input"),
          py::arg("other"),
          R"doc(Computes the element-wise minimum of input and other.)doc");

    m.def("minimum_",
          &op::minimum_,
          py::arg("input"),
          py::arg("other"),
          py::arg("output"),
          R"doc(In-place version of minimum.)doc");
}

} // namespace infinicore::ops