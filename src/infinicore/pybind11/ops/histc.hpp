#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/histc.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_histc(py::module &m) {
    m.def("histc",
          &op::histc,
          py::arg("input"),
          py::arg("bins"),
          py::arg("min"),
          py::arg("max"),
          R"doc(Computes the histogram of a tensor.)doc");

    m.def("log10_",
          &op::histc_,
          py::arg("input"),
          py::arg("output"),
          py::arg("bins"),
          py::arg("min"),
          py::arg("max"),
          R"doc(In-place histogram computation.)doc");
}

} // namespace infinicore::ops
