#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/maximum.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_maximum(py::module &m) {
    m.def("maximum",
          &op::maximum,
          py::arg("a"),
          py::arg("b"),
          R"doc(Element-wise maximum of two tensors.)doc");

    m.def("maximum_",
          &op::maximum_,
          py::arg("c"),
          py::arg("a"),
          py::arg("b"),
          R"doc(In-place element-wise tensor maximum.)doc");
}

} // namespace infinicore::ops
