#pragma once

#include "infinicore/ops/sqrt.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {
inline void bind_sqrt(py::module &m) {
    m.def("sqrt", &op::sqrt, py::arg("input"),
          R"doc(Element-wise square root.)doc");
    m.def("sqrt_", &op::sqrt_, py::arg("output"), py::arg("input"),
          R"doc(In-place square root.)doc");
}
} // namespace infinicore::ops