#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/logsumexp.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_logsumexp(py::module &m) {
    m.def("logsumexp",
          &op::logsumexp,
          py::arg("input"),
          py::arg("dim"),
          py::arg("keepdim"),
          R"doc(Logarithm of the sum of exponentials of the input tensor.)doc");

    m.def("logsumexp_",
          &op::logsumexp_,
          py::arg("input"),
          py::arg("dim"),
          py::arg("keepdim"),
          py::arg("output"),
          R"doc(In-place logarithm of the sum of exponentials of the input tensor.)doc");
}

} // namespace infinicore::ops
