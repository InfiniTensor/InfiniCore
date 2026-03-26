#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/round.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_round(py::module &m) {
    m.def("round",
          &op::round,
          py::arg("x"),
          py::arg("decimals") = 0,
          R"doc(Element-wise rounding with optional decimal places.)doc");

    m.def("round_",
          &op::round_,
          py::arg("y"),
          py::arg("x"),
          py::arg("decimals") = 0,
          R"doc(In-place element-wise rounding.)doc");
}

} // namespace infinicore::ops