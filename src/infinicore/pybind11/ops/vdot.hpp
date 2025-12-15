#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/vdot.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_vdot(py::module &m) {
    m.def("vdot",
          &op::vdot,
          py::arg("a"),
          py::arg("b"),
          R"doc(Vector dot product for 1D tensors (real-valued).)doc");
}

} // namespace infinicore::ops


