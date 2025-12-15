#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/log2.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_log2(py::module &m) {
    m.def("log2",
          &op::log2,
          py::arg("input"),
          R"doc(Logarithm base 2 of the tensor.)doc");

    m.def("log2_",
          &op::log2_,
          py::arg("input"),
          py::arg("output"),
          R"doc(In-place logarithm base 2 computation.)doc");
}

} // namespace infinicore::ops