#pragma once

#include "infinicore/ops/atan2.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace infinicore::ops {

inline void bind_atan2(py::module &m) {
    m.def("atan2",
          &op::atan2,
          py::arg("input"),
          py::arg("other"),
          R"doc(Computes the element-wise arc tangent of input/other. Returns the angle in radians.)doc");

    m.def("atan2_",
          &op::atan2_,
          py::arg("input"),
          py::arg("other"),
          py::arg("output"),
          R"doc(In-place version of atan2.)doc");
}

} // namespace infinicore::ops