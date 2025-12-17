#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/log1p.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_log1p(py::module &m) {
    m.def("log1p",
          &op::log1p,
          py::arg("input"),
          R"doc(Returns a new tensor with the natural logarithm of (1 + input).)doc");

    m.def("log1p_",
          &op::log1p_,
          py::arg("input"),
          py::arg("output"),
          R"doc(In-place computation of the natural logarithm of (1 + input).)doc");
}

} // namespace infinicore::ops
