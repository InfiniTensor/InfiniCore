#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/log10.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_log10(py::module &m) {
    m.def("log10",
          &op::log10,
          py::arg("input"),
          R"doc(Logarithm base 10 of the tensor.)doc");

    m.def("log10_",
          &op::log10_,
          py::arg("input"),
          py::arg("output"),
          R"doc(In-place logarithm base 10 computation.)doc");
}

} // namespace infinicore::ops
