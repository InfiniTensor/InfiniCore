#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/erfinv.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_erfinv(py::module &m) {
    m.def("erfinv",
          &op::erfinv,
          py::arg("input"),
          R"doc(Inverse error function (erfinv) of input tensor.)doc");

    m.def("erfinv_",
          &op::erfinv_,
          py::arg("output"),
          py::arg("input"),
          R"doc(In-place inverse error function (erfinv) of input tensor.)doc");
}

} // namespace infinicore::ops