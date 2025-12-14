#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/erf.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_erf(py::module &m) {
    m.def("erf",
          &op::erf,
          py::arg("input"),
          R"doc(Error function (erf) of input tensor.)doc");

    m.def("erf_",
          &op::erf_,
          py::arg("output"),
          py::arg("input"),
          R"doc(In-place error function (erf) of input tensor.)doc");
}

} // namespace infinicore::ops