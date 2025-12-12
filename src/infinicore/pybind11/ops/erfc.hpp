#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/erfc.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_erfc(py::module &m) {
    m.def("erfc",
          &op::erfc,
          py::arg("input"),
          R"doc(Complementary error function (erfc) of input tensor.)doc");

    m.def("erfc_",
          &op::erfc_,
          py::arg("output"),
          py::arg("input"),
          R"doc(In-place complementary error function (erfc) of input tensor.)doc");
}

} // namespace infinicore::ops