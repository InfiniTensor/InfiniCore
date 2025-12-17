#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/avg_pool3d.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_avg_pool3d(py::module &m) {
    m.def("avg_pool3d",
          &op::avg_pool3d,
          py::arg("input"),
          py::arg("kernel_size"),
          py::arg("stride") = py::none(),
          py::arg("padding") = 0,
          py::arg("ceil_mode") = false,
          R"doc(Applies 3D average-pooling operation in :math:`kD \ times kH \times kW` regions by step size
    :math:`sD \times sH \times sW` steps. The number of output features is equal to the number of
    input planes.)doc");
}

} // namespace infinicore::ops
