#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/lp_pool3d.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_lp_pool3d(py::module &m) {
    m.def("lp_pool3d",
          &op::lp_pool3d,
          py::arg("input"),
          py::arg("norm_type"),
          py::arg("kernel_size"),
          py::arg("stride") = py::none(),
          py::arg("ceil_mode") = false,
          R"doc(Applies a 3D power-average pooling over an input signal composed of several input planes.)doc");
}

} // namespace infinicore::ops
