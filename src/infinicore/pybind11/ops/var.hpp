#pragma once

#include "infinicore/ops/var.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace infinicore::ops {

inline void bind_var(py::module &m) {
    m.def("var_global",
          &op::var_global,
          py::arg("input"),
          py::arg("correction") = 1.0,
          R"doc(Global variance.)doc");

    m.def("var_global_",
          &op::var_global_,
          py::arg("input"),
          py::arg("output"),
          py::arg("correction") = 1.0,
          R"doc(In-place global variance.)doc");

    m.def("var_reduce",
          &op::var_reduce,
          py::arg("input"),
          py::arg("dim"),
          py::arg("correction") = 1.0,
          py::arg("keepdim") = false,
          R"doc(Variance reduction along dim.)doc");

    m.def("var_reduce_",
          &op::var_reduce_,
          py::arg("input"),
          py::arg("output"),
          py::arg("dim"),
          py::arg("correction") = 1.0,
          py::arg("keepdim") = false,
          R"doc(In-place variance reduction.)doc");
}

} // namespace infinicore::ops