#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/dot.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_dot(py::module &m) {
    m.def("dot",
          &op::dot,
          py::arg("input"),
          py::arg("tensor"),
          R"doc(Computes the dot product of two 1-D tensors.)doc");
}

} // namespace infinicore::ops
