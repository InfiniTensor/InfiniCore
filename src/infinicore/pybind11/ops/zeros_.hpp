#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/zeros.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_zeros_(py::module &m) {
    m.def("zeros_",
          &op::zeros_,
          py::arg("input"),
          R"doc(Fills the input tensor with zeros.)doc");
}

} // namespace infinicore::ops
