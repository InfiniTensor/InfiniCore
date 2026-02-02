#pragma once

#include "infinicore/ops/nll_loss.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace infinicore::ops {

inline void bind_nll_loss(py::module &m) {
    m.def("nll_loss",
          &op::nll_loss,
          py::arg("input"),
          py::arg("target"),
          py::arg("weight") = py::none(),
          py::arg("ignore_index") = -100,
          R"doc(Calculates NLL Loss.)doc");

    m.def("nll_loss_",
          &op::nll_loss_,
          py::arg("input"),
          py::arg("target"),
          py::arg("weight"),
          py::arg("output"),
          py::arg("ignore_index") = -100,
          R"doc(In-place NLL Loss calculation.)doc");
}

} // namespace infinicore::ops