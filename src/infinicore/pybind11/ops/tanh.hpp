#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/tanh.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_tanh(py::module &m) {
    m.def("tanh",
          &op::tanh,
          py::arg("input"),
          R"doc(Out-of-place hyperbolic tangent.)doc");

    m.def("tanh_",
          &op::tanh_,
          py::arg("output"),
          py::arg("input"),
          R"doc(Hyperbolic tangent writing to the provided output.)doc");
}

} // namespace infinicore::ops
