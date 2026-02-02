#pragma once
#include "infinicore/ops/glu.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace infinicore::ops {

inline void bind_glu(py::module &m) {
    m.def("glu",
          &op::glu,
          py::arg("input"),
          py::arg("dim") = -1);

    m.def("glu_",
          &op::glu_,
          py::arg("input"),
          py::arg("output"),
          py::arg("dim") = -1);
}

} // namespace infinicore::ops